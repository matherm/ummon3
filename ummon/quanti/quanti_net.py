# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2018-12-21 17:11:17
# @Last Modified by:   daniel
# @Last Modified time: 2019-04-12 18:06:36

import torch
import torch.nn as nn
from .quantization import Quantization, stdParam
import logging
import copy
import importlib
import sys


##
# @brief      Class adding quantization function to a model.
##
class QuantiNetWrapper(nn.Module):
    ##
    # @brief      Constructs the object. #
    #
    # @param      self                     The object
    # @param      wrapping_net             can be passed as module name, as
    #                                      string of module name, or as an
    #                                      instance
    # @param      use_quantization         apply quantization (However can be
    #                                      done later)
    # @param      quanti_param_net         The quantization-parameters for model
    #                                      activations
    # @param      quanti_param_parameters  The quantization-parameter for model
    #                                      weights
    # @param      supported_container      The supported container type
    # @param      modules_for_quanti       A list of modules after quantization
    #                                      layer will be added
    # @param      replace_modules          A dict of modules which must be
    #                                      replaced. Formate: {replace_type:
    #                                      replace_function(module, qp_net)}
    # @param      net_args                 Some model arguments
    # @param      net_kwargs               Some model kwargs #
    #
    def __init__(
        self,
        wrapping_net,
        use_quantization=False,
        quanti_param_net=stdParam(8),
        quanti_param_parameters=stdParam(8),
        supported_container=[nn.Sequential],
        modules_for_quanti=[nn.Conv2d, nn.ReLU, nn.Linear],
        replace_modules={},
        *net_args,
        **net_kwargs
    ):
        super().__init__()
        self.__log = logging.getLogger('quanti_log')
        if type(wrapping_net) is type:
            self.net = wrapping_net(*net_args, **net_kwargs)
        elif type(wrapping_net) is str:
            module, classname = wrapping_net.rsplit('.', 1)
            importlib.import_module(module)
            __class = getattr(sys.modules[module], classname)
            self.net = __class(*net_args, **net_kwargs)
        else:
            self.net = wrapping_net
        self.use_quantization = use_quantization
        self.quantization_f = None
        self.__quanti_params = {}
        self.__supported_container = supported_container
        self.__modules_for_quanti = modules_for_quanti
        self.__replace_modules = replace_modules
        if self.use_quantization:
            self.add_and_perform_quantization(
                quanti_param_net,
                quanti_param_parameters)

    def forward(self, x):
        if self.use_quantization:
            self.__quantizeParamsInForward()
        return self.net(x)

    ##
    # @brief      Adds and perform quantization. Function can be used to activate quantization at a later point
    ##
    # @param      self           The object
    # @param      qp_net         The quantization-parameters for model activations
    # @param      qp_parameters  The quantization-parameter for model weights
    ##
    # @return     None
    ##
    def add_and_perform_quantization(self, qp_net, qp_parameters):
        self.use_quantization = True
        self.__quanti_params = dict(qp_net=qp_net, qp_parameters=qp_parameters)
        self.__add_quantization(qp_net)
        self.__quantizeParams(qp_parameters)
        self.__log.debug(self)

    ##
    # @brief      Removes the quantization functions.
    ##
    # @param      self  The object
    ##
    # @return     None
    ##
    def remove_quantization(self):
        self.use_quantization = False
        delattr(self, 'quantization_f')
        self.quantization_f = None
        self.__quanti_params = {}

        for container in self.__finde_containers():
            con = type(container['c'])()
            for i, (name, module) in enumerate(container['c'].named_modules()):
                if not (type(module) in self.__supported_container):
                    if type(module) is not Quantization:
                        con.add_module(name, module)
            setattr(self.net, container['name'], con)

    def get_qunati_params(self):
        return copy.deepcopy(self.__quanti_params)

    def __finde_containers(self):
        c_list = []
        for n, m in self.net.named_modules():
            if type(m) in self.__supported_container:
                c_list.append(dict(name=n, c=m))
        if len(c_list) == 0:
            self.__log.error("no module conatiner found")
        self.__log.debug("Found {} containers".format(len(c_list)))
        return c_list

    def __add_quantization_r(self, module, res_container, qp_net):
        for name, _module in module.named_children():
            # if sum(1 for _ in _module.children()) != 0:
            #     _module = self.__add_quantization_r(_module, nn.Sequential(), qp_net)
            if type(_module) in self.__replace_modules.keys():
                _module = self.__replace_modules[type(
                    _module)](_module, qp_net)
            res_container.add_module(name, _module)
            if (type(_module) in self.__modules_for_quanti):
                self.__idx += 1
                res_container.add_module(
                    "quanti_" + str(self.__idx), Quantization(qp_net))

        return res_container

    def __add_quantization(self, qp_net):
        self.__idx = 0
        for container in self.__finde_containers():
            # create module container
            con = type(container['c'])()
            con.add_module("quanti", Quantization(qp_net))

            # loop trough child modules
            con = self.__add_quantization_r(container['c'], con, qp_net)

            setattr(self.net, container['name'], con)

    def __quantizeParamsInForward(self):
        """
        @brief      apply quantization on net parameters without distribution shifting.
                    This function changed the net parameters
        @param      self  The object
        @return     None
        """
        for param in self.net.parameters():
            param.data = self.quantization_f(param.data)

    def __quantizeParams(self, qp_for_netparams):
        """
        @brief      apply quantization on net parameters with the given distribution shifting.
                    This function changed the net parameters.
                    It also save a quantization module which is called every time while forward function is executed.
        @param      self  The object
        @return     None
        """
        # save quatnization fuction for forwarding
        param_forward = stdParam(bitwith=qp_for_netparams.bitwith)
        self.quantization_f = Quantization(param_forward)

        q = Quantization(qp_for_netparams)
        for param in self.net.parameters():
            param.data = q(param.data)
            # used in VA script
            param.quanti_param = qp_for_netparams
