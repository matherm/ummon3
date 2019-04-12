# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2019-03-19 22:37:31
# @Last Modified by:   Daniel
# @Last Modified time: 2019-04-11 16:10:58
import pytest
import torch
from torch import nn
from .quantization import stdParam
from .quanti_net import QuantiNetWrapper
import os
import logging

logging.getLogger().setLevel(logging.NOTSET)

# RUN python -m pytest quanti


class MyTestNet(nn.Module):
    # define Layers
    def __init__(self):
        super().__init__()
        self.convol = torch.nn.Sequential()
        self.convol.add_module("conv_1", nn.Conv2d(
            1, 6, 5, padding=2))
        self.convol.add_module("maxpool_1", nn.MaxPool2d(2))
        self.convol.add_module("ReLu_1", nn.ReLU())

        self.convol.add_module("conv_2", nn.Conv2d(
            6, 16, 5, padding=2))
        self.convol.add_module("maxpool_2", nn.MaxPool2d(2))
        self.convol.add_module("ReLu_2", nn.ReLU())

        self.fc = torch.nn.Sequential()
        self.fc.add_module("fc_1", nn.Linear(
            int(16 * (((28 - 0) / 2 - 0) / 2) ** 2), 120))
        # self.fc.add_module("fc_1", nn.Linear(16 * 4 * 4, 120))
        self.fc.add_module("fc_ReLu_3", nn.ReLU())

        self.fc.add_module("fc_2", nn.Linear(120, 84))
        self.fc.add_module("fc_ReLu_4", nn.ReLU())

        self.fc.add_module("fc_3", nn.Linear(84, 10))

    # connect Layers
    def forward(self, x):
        x = self.conv.forward(x)  # apply convolution
        # reshape for fully connected
        x = x.view(-1, self.num_flat_features(x))
        return self.fc.forward(x)  # apply Linear functions

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class TestQuantizeForwardpath(object):
    ver_net = MyTestNet()
    quanti_param = {
        "qp_net": stdParam(8, divisor=1),
        "qp_parameters": stdParam(8, divisor=8)
    }
    __log = logging.getLogger('quanti_log')

    def test_instance(self):
        net = QuantiNetWrapper(wrapping_net=MyTestNet())

        torch.save(net.state_dict(), "./__MyTestNet.pth")

        self.__log.debug("net befor adding qunatization: {}".format(net))
        net.add_and_perform_quantization(**self.quanti_param)
        net.remove_quantization()
        self.__log.debug("original net (should be the same as before adding quanti): {}".format(net))

        assert str(net.net) == str(self.ver_net), "net differs "
        net.load_state_dict(torch.load('./__MyTestNet.pth'))
        os.remove('./__MyTestNet.pth')

    def test_type(self):
        net = QuantiNetWrapper(wrapping_net=MyTestNet)

        torch.save(net.state_dict(), "./__MyTestNet.pth")

        self.__log.debug("net befor adding qunatization: {}".format(net))
        net.add_and_perform_quantization(**self.quanti_param)
        net.remove_quantization()
        self.__log.debug("original net (should be the same as before adding quanti): {}".format(net))

        assert str(net.net) == str(self.ver_net), "net differs "
        net.load_state_dict(torch.load('./__MyTestNet.pth'))
        os.remove('./__MyTestNet.pth')

    def test_string(self):
        net = QuantiNetWrapper(wrapping_net="quanti.test_quanti.MyTestNet")

        torch.save(net.state_dict(), "./__MyTestNet.pth")

        self.__log.debug("net befor adding qunatization: {}".format(net))
        net.add_and_perform_quantization(**self.quanti_param)
        net.remove_quantization()
        self.__log.debug("original net (should be the same as before adding quanti): {}".format(net))

        assert str(net.net) == str(self.ver_net), "net differs "
        net.load_state_dict(torch.load('./__MyTestNet.pth'))
        os.remove('./__MyTestNet.pth')
