# -*- coding: utf-8 -*-
# @Author: Daniel
# @Date:   2019-05-13 16:10:48
# @Last Modified by:   Daniel
# @Last Modified time: 2019-05-14 15:20:19
#--------------------------------------------------------------------
# Downloads the model from iosds02-NAS
import os
from ummon.utils.sftp_helpers import SFTP
server_path = "/Installer/pretrained-models/pytorch_mobilenet_params/mobilenet_sgd_rmsprop_69.526.tar"
model_path = "~/.cache/mobilenet/mobilenet_sgd_rmsprop_69.526.tar"
if not os.path.exists(model_path):
    os.makedirs(model_path[:-1])
    print("Downloading pretrained Mobilenet from {} to {}.".format(server_path, model_path))
    sftp = SFTP(host="iosds02.ios.htwg-konstanz.de", port=22, user="ios-dataset-user", password="ios123")
    sftp.get(src=server_path, dest=model_path)
#--------------------------------------------------------------------

import torch
import torch.nn as nn
from collections import OrderedDict
from ummon.modules.gram import GramMatrix, GramDiag
import copy
import os.path as path
from collections import OrderedDict

# implementation from https://github.com/marvis/pytorch-mobilenet


class Reshape(nn.Module):
    def __init__(self, nb_flatt_features):
        super().__init__()
        self.nb_flatt_features = nb_flatt_features

    def forward(self, x):
        return x.view(-1, self.nb_flatt_features)

    def __repr__(self):
        return __class__.__name__ + " (nb_flatt_features={})".format(self.nb_flatt_features)


class MobileNet(nn.Module):

    def __init__(self,
                 version=1,
                 num_classes=1000,
                 pretrained=False,
                 layer="",
                 gram=False,
                 gram_diag=False,
                 gram_diagonal_squared=False,
                 params_file=model_path):
        super().__init__()
        if version not in [1]:
            raise ValueError("Unsupported MobileNet version {version}:"
                             "1 expected".format(version=version))

        self.num_classes = num_classes
        # build layer names for selection
        ref_mobile_net = None
        if version == 1:
            ref_mobile_net = marvisMobileNet(pretrained,
                                             params_file=params_file,
                                             num_classes=num_classes)
            features_names = OrderedDict(
                conv=5,

                depthwise_conv_1=5 + 4,
                pointwise_conv_1=5 + 7,

                depthwise_conv_2=5 + 1 * 7 + 4,
                pointwise_conv_2=5 + 1 * 7 + 7,

                depthwise_conv_3=5 + 2 * 7 + 4,
                pointwise_conv_3=5 + 2 * 7 + 7,

                depthwise_conv_4=5 + 3 * 7 + 4,
                pointwise_conv_4=5 + 3 * 7 + 7,

                depthwise_conv_5=5 + 4 * 7 + 4,
                pointwise_conv_5=5 + 4 * 7 + 7,

                depthwise_conv_6=5 + 5 * 7 + 4,
                pointwise_conv_6=5 + 5 * 7 + 7,

                depthwise_conv_7=5 + 6 * 7 + 4,
                pointwise_conv_7=5 + 6 * 7 + 7,

                depthwise_conv_8=5 + 7 * 7 + 4,
                pointwise_conv_8=5 + 7 * 7 + 7,

                depthwise_conv_9=5 + 8 * 7 + 4,
                pointwise_conv_9=5 + 8 * 7 + 7,

                depthwise_conv_10=5 + 9 * 7 + 4,
                pointwise_conv_10=5 + 9 * 7 + 7,

                depthwise_conv_11=5 + 10 * 7 + 4,
                pointwise_conv_11=5 + 10 * 7 + 7,

                depthwise_conv_12=5 + 11 * 7 + 4,
                pointwise_conv_12=5 + 11 * 7 + 7,

                depthwise_conv_13=5 + 12 * 7 + 4,
                pointwise_conv_13=5 + 12 * 7 + 7,

                avg_pool=5 + 13 * 7 + 1,

                fc=5 + 13 * 7 + 2
            )

        assert layer in features_names.keys(),\
            "layername not supported. Supported names: {}".format(
                features_names.keys())
        # copy selected squeezenet features
        self.features = torch.nn.Sequential()

        keys = list(features_names.keys())
        keys_idx = 0
        level_name = keys[keys_idx]
        container = nn.Sequential()
        self.features.add_module(level_name, container)
        first = True

        for i, (name, module) in enumerate(ref_mobile_net.named_modules()):
            if i > features_names[keys[keys_idx]]:
                keys_idx += 1
                level_name = keys[keys_idx]
                container = nn.Sequential()
                self.features.add_module(level_name, container)
                if first and level_name is "fc":
                    container.add_module("Reshape", Reshape(1024))
                    first = False

            mname = level_name + "_" + name.replace('.', '_')
            if type(module) is not nn.Sequential and \
                    type(module) is not marvisMobileNet:
                container.add_module(mname, copy.deepcopy(module))

            if i == features_names[layer]:
                break

        del ref_mobile_net
        # add gram calculation?
        if gram:
            self.features.add_module("gram matrix", GramMatrix())
        elif gram_diag:
            self.features.add_module(
                "gram diagonal", GramDiag(gram_diagonal_squared))

    def forward(self, x):
        return self.features(x)


class marvisMobileNet(nn.Module):
    def __init__(self, pretrained=False, params_file="", num_classes=1000):
        super().__init__()
        self.pretrained = pretrained

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True)
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),

                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        self.model = nn.Sequential(
            conv_bn(3, 32, 2),
            conv_dw(32, 64, 1),
            conv_dw(64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, num_classes)

        if self.pretrained:
            # w = torch.load("pytorch-mobilenet/mobilenet_sgd_rmsprop_69.526.tar")
            # self.load_state_dict(w)

            # https://github.com/marvis/pytorch-mobilenet/issues/14
            tar = torch.load(params_file)
            state_dict = tar['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x


def main():
    model = MobileNet(pretrained=True, layer="pointwise_conv_2")
    print(model)
    # print(marvisMobileNet(pretrained=True))


if __name__ == '__main__':
    main()
