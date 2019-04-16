# -*- coding: utf-8 -*-
# @Author: Daniel
# @Date:   2019-04-10 14:30:56
# @Last Modified by:   Daniel
# @Last Modified time: 2019-04-16 13:37:42
from torchvision.models.squeezenet import squeezenet1_0, squeezenet1_1
import torch
import torch.nn as nn
import copy
from ummon.modules.gram import GramMatrix, GramDiag


class SqueezeNet(nn.Module):

    def __init__(self, version=1.0, num_classes=1000, pretrained=False, layer="", gram=False, gram_diag=False, gram_diagonal_squared=False):
        super().__init__()
        if version not in [1.0, 1.1]:
            raise ValueError("Unsupported SqueezeNet version {version}:"
                             "1.0 or 1.1 expected".format(version=version))
        self.num_classes = num_classes
        # build layer names for selection
        if version == 1.0:
            pytorch_squeeze = squeezenet1_0(
                pretrained, num_classes=num_classes)
            features_names = [
                "conv_1",
                "relu_1",
                "maxpool_1",
                "fire_2",
                "fire_3",
                "fire_4",
                "maxpool_4",
                "fire_5",
                "fire_6",
                "fire_7",
                "fire_8",
                "maxpool_8",
                "fire_9"]
        else:
            pytorch_squeeze = squeezenet1_1(
                pretrained, num_classes=num_classes)
            features_names = [
                "conv_1",
                "relu_1",
                "maxpool_1",
                "fire_2",
                "fire_3",
                "maxpool_3",
                "fire_4",
                "fire_5",
                "maxpool_5",
                "fire_6",
                "fire_7",
                "fire_8",
                "fire_9"]
        # Final convolution is initialized differently form the rest
        classifier_names = [
            "drop_10",
            "conv_10",
            "relu_10",
            "avgpool_10"]

        # copy selected squeezenet features
        self.features = torch.nn.Sequential()
        for name, module in zip(features_names, pytorch_squeeze.features):
            self.features.add_module(name, copy.deepcopy(module))
            if layer is name:
                break
        # copy the clasifier layer?
        if len(features_names) == len(self.features) and layer != features_names[-1]:
            for name, module in zip(classifier_names, pytorch_squeeze.classifier):
                self.features.add_module(name, copy.deepcopy(module))
                if layer is name:
                    break
        del(pytorch_squeeze)

        # add gram calculation?
        if gram:
            self.features.add_module("gram matrix", GramMatrix())
        elif gram_diag:
            self.features.add_module(
                "gram diagonal", GramDiag(gram_diagonal_squared))

    def forward(self, x):
        return self.features(x)


def test():
    print(SqueezeNet(pretrained=True, layer="fire_7"))
    print(SqueezeNet(pretrained=True, layer="fire_9"))
    print(SqueezeNet(pretrained=True, layer="conv_10"))


if __name__ == '__main__':
    test()
