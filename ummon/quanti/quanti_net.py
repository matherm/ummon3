# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2018-12-21 17:11:17
# @Last Modified by:   Daniel
# @Last Modified time: 2019-04-11 18:27:25

import torch
import torch.nn as nn
from .quantization import Quantization, stdParam
import logging
import copy
import importlib
import sys

class QuantiNetWrapper(nn.Module):
    # wrapping_net: can be passed as module name, as string of module name, or as an instance
    def __init__(
        self,
        wrapping_net,
        use_quantization=False,
        quanti_param_net=stdParam(8),
        quanti_param_parameters=stdParam(8),
        modules_for_quanti=[nn.Conv2d, nn.ReLU, nn.Linear],
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
        self.supported_container = [torch.nn.Sequential]
        self.supported_container += [torch.nn.ModuleList]
        self.__modules_for_quanti = modules_for_quanti
        if self.use_quantization:
            self.add_and_perform_quantization(
                quanti_param_net,
                quanti_param_parameters)

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        if self.use_quantization:
            self.__quantizeParamsInForward()
        return self.net(x)

    def add_and_perform_quantization(self, qp_net, qp_parameters):
        self.use_quantization = True
        self.__quanti_params = dict(qp_net=qp_net, qp_parameters=qp_parameters)
        self.__add_quantization(qp_net)
        self.__quantizeParams(qp_parameters)
        self.__log.debug(self)

    def remove_quantization(self):
        self.use_quantization = False
        delattr(self, 'quantization_f')
        self.quantization_f = None
        self.__quanti_params = {}

        for container in self.__finde_containers():
            con = type(container['c'])()
            for i, (name, module) in enumerate(container['c'].named_modules()):
                if not (type(module) in self.supported_container):
                    if type(module) is not Quantization:
                        con.add_module(name, module)
            setattr(self.net, container['name'], con)

    def __finde_containers(self):
        c_list = []
        for n, m in self.net.named_modules():
            if type(m) in self.supported_container:
                c_list.append(dict(name=n,c=m))
        if len(c_list) == 0:
            self.__log.error("no module conatiner found")
        self.__log.debug("Found {} containers".format(len(c_list)))
        return c_list

    # qp := quantization parameter
    def __add_quantization(self, qp_net):
        for container in self.__finde_containers():
            con = type(container['c'])()
            con.add_module("quanti", Quantization(qp_net))
            for i, (name, module) in enumerate(container['c'].named_modules()):
                if not (type(module) in self.supported_container):
                    print("name: {}, type{}".format(name, module))
                    con.add_module(name, module)
                    if (type(module) in self.__modules_for_quanti):
                        con.add_module("quanti_"+str(i), Quantization(qp_net))
            setattr(self.net, container['name'], con)

    def __add_quantization_r(self, module, res_container):
        for i, (name, module) in enumerate(module.named_modules()):
            if not (type(module) in self.supported_container):
                if sum(1 for _ in module.children()) != 0:
                    module = self.__add_quantization_r(module,module)
                res_container.add_module(name, module)
                if (type(module) in self.__modules_for_quanti):
                    res_container.add_module("quanti_"+str(i), Quantization(qp_net))

        return res_container


    def __add_quantization(self, qp_net):
        for container in self.__finde_containers():
            con = type(container['c'])()
            con.add_module("quanti", Quantization(qp_net))


            self.__add_quantization_r(container['c'])
            
            for i, (name, module) in enumerate(container['c'].named_modules()):
                if not (type(module) in self.supported_container):
                    print("name: {}, type{}".format(name, module))
                    con.add_module(name, module)
                    if (type(module) in self.__modules_for_quanti):
                        con.add_module("quanti_"+str(i), Quantization(qp_net))


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

    def get_qunati_params(self):
        return copy.deepcopy(self.__quanti_params)
