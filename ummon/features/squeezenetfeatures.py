# -*- coding: utf-8 -*-
# @Author: Daniel
# @Date:   2019-04-16 10:37:18
# @Last Modified by:   Daniel
# @Last Modified time: 2019-04-16 15:53:20
import torch
import torch.nn as nn
from ummon.modules.squeezenet import SqueezeNet
from ummon.quanti.quanti_net import QuantiNetWrapper
from ummon.quanti.quantization import stdParam, mean_std
from ummon.quanti.spezial_qunati_modules import replace_fire_module
from ummon.modules.gram import GramDiag, GramMatrix
from torchvision.models.squeezenet import Fire
import logging


class SqueezeNetFeatures():
    """
    Extracts features from SqueezeNet

    Usage
    ======
            transform = SqueezeNet(features="fire_4")
            transform(tensor)

            OR

            vgg = SqueezeNetFeatures(features="fire_4")
            my_transforms = transforms.Compose([transforms.ToTensor(), vgg])
            test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms)

    Input
    ======
        *tensor (torch.Tensor) with shape (B x 3 x min 32 x 32)

    Return
    =====
        *feature (torch.Tensor)

    """

    ##
    # @brief      Constructs the object.
    ##
    # @param      self                   The object
    # @param      features               Selects the last layer to use
    # @param      cuda                   (bool) using cuda
    # @param      pretrained             Use pretrained SqueezeNet
    # @param      gram                   calculate features using gram matrix
    # @param      gram_diagonal          calculate features using gram
    # diagnonal
    # @param      gram_diagonal_squared  calculate features using gram
    # diagnonal^4
    # @param      version                SqueezeNet version (supported [1.0,
    # 1.1])
    ##
    def __init__(self,
                 features="fire4",
                 cuda=False,
                 pretrained=True,
                 gram=False,
                 gram_diagonal=False,
                 gram_diagonal_squared=False,
                 version=1.0):
        self.__cuda = cuda
        self.model = SqueezeNet(layer=features,
                                pretrained=pretrained,
                                gram=gram,
                                gram_diag=gram_diagonal,
                                gram_diagonal_squared=gram_diagonal_squared,
                                version=version)

        if cuda and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.model = self.model.eval()

    def __call__(self, x):
        if x.dim != 3 and x.size(0) != 3:
            raise ValueError(
                "Shape of input should be (3, x, x), is:", x.shape)
        x = x.unsqueeze(0)

        if self.__cuda and torch.cuda.is_available():
            x = x.to('cuda')

        feature = self.model(x)
        feature = feature.detach().to('cpu').view(-1).contiguous()
        return feature


class SqueezeNetFeaturesQuanti():
    """
    Extracts features from SqueezeNet

    Usage
    ======
            transform = SqueezeNet(features="fire_4")
            transform(tensor)

            OR

            vgg = SqueezeNetFeatures(features="fire_4")
            my_transforms = transforms.Compose([transforms.ToTensor(), vgg])
            test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms)

    Input
    ======
        *tensor (torch.Tensor) with shape (B x 3 x min 32 x 32)

    Return
    =====
        *feature (torch.Tensor)

    """

    ##
    # @brief      Constructs the object. #
    #
    # @param      self                   The object
    # @param      features               Selects the last layer to use
    # @param      cuda                   (bool) using cuda
    # @param      pretrained             Use pretrained SqueezeNet
    # @param      gram                   calculate features using gram matrix
    # @param      gram_diagonal          calculate features using gram diagnonal
    # @param      gram_diagonal_squared  calculate features using gram
    #                                    diagnonal^4
    # @param      version                SqueezeNet version (supported [1.0,
    #                                    1.1]) #
    # @param      bits                   bitlength for quantization
    # @param      nb_sigma               The number of sigma for distribution
    #                                    shifting
    #
    def __init__(self,
                 features="fire4",
                 cuda=False,
                 pretrained=True,
                 gram=False,
                 gram_diagonal=False,
                 gram_diagonal_squared=False,
                 version=1.0,
                 bits=8,
                 nb_sigma=1):
        self.__cuda = cuda
        self.model = SqueezeNet(layer=features,
                                pretrained=pretrained,
                                gram=gram,
                                gram_diag=gram_diagonal,
                                gram_diagonal_squared=gram_diagonal_squared,
                                version=version)
        self.bits = bits
        self.nb_sigma = nb_sigma
        self.gram = gram
        self.gram_diagonal = gram_diagonal
        self.gram_diagonal_squared = gram_diagonal_squared
        self.features = features
        self.pretrained = pretrained

        if cuda and torch.cuda.is_available():
            self.model = self.model.to('cuda')
        self.model = self.model.eval()

        dist = 1 / 2**(bits - 1)
        _max = 1 - (dist * 1)
        _min = -1 + (dist * 1)
        clip_range = dict(min=_min, max=_max)
        clip_range_n = dict(min=-1, max=_max)
        mean, std = mean_std(self.model.parameters())
        logging.getLogger().info("Parameters mean: {}, parameters std: {}".format(mean, std))
        params = {
            "qp_net": stdParam(bits,
                               divisor=1,
                               handle_overflow=False,
                               rounding=False,
                               clip_range=clip_range_n),
            "qp_parameters": stdParam(bits,
                                      divisor=nb_sigma,
                                      mean=mean,
                                      std=std,
                                      handle_overflow=False,
                                      clip_range=clip_range)}
        self.model = QuantiNetWrapper(
            self.model,
            replace_modules={Fire: replace_fire_module},
            modules_for_quanti=[nn.Conv2d, nn.Linear, GramMatrix, GramDiag])
        self.model.add_and_perform_quantization(**params)

    def __call__(self, x):
        if x.dim != 3 and x.size(0) != 3:
            raise ValueError(
                "Shape of input should be (3, x, x), is:", x.shape)
        x = x.unsqueeze(0)

        if self.__cuda and torch.cuda.is_available():
            x = x.to('cuda')

        feature = self.model(x)
        feature = feature.detach().to('cpu').view(-1).contiguous()
        return feature
