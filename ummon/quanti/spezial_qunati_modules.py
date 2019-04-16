# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2019-04-12 12:18:55
# @Last Modified by:   daniel
# @Last Modified time: 2019-04-12 18:19:29

import torch
from torchvision.models.squeezenet import Fire
from .quantization import Quantization
import copy


##
# @brief      Helper module for the fire layer used by squeezeNet. Originally
# taken from: torchvision.models.squeezenet
##
class FireCopyQunatize(torch.nn.Module):
    ##
    # @brief      Constructs the object.
    ##
    # @param      self    The object
    # @param      fire    The fire
    # @param      qp_net  The qp net
    ##
    def __init__(self, fire, qp_net):
        super().__init__()
        self.squeeze = copy.deepcopy(fire.squeeze)
        self.squeeze_activation = copy.deepcopy(fire.squeeze_activation)
        self.expand1x1 = copy.deepcopy(fire.expand1x1)
        self.expand1x1_activation = copy.deepcopy(fire.expand1x1_activation)
        self.expand3x3 = copy.deepcopy(fire.expand3x3)
        self.expand3x3_activation = copy.deepcopy(fire.expand3x3_activation)
        self.quantize = Quantization(qp_net)

    def forward(self, x):
        x = self.squeeze_activation(self.quantize(self.squeeze(x)))
        return torch.cat([
            self.expand1x1_activation(self.quantize(self.expand1x1(x))),
            self.expand3x3_activation(self.quantize(self.expand3x3(x)))
        ], 1)


##
# @brief      Helperfunction for quanti net wrapper to replace the fire module.
#             (weights are copyed from input module) #
#
# @param      module  The module (type Fire)
# @param      qp_net  The qunatization-parameters for the activations #
#
# @return     a new fire module which performs quantization #
#
def replace_fire_module(module: Fire, qp_net):
    assert type(module) is Fire, \
        "module is not of type torchvision.models.squeezenet.Fire"
    return FireCopyQunatize(module, qp_net)
