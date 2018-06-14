#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:42:00 2018


Custom Implementation of the VGG19 Network

@author: Fabian Freiberg
"""

__all__ = [ 'VGG19']

import torch.nn as nn

class VGG19(nn.Module):
    def __init__(self, pool='max'):
        super(VGG19, self).__init__()

        # W x H x 3

        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU()
        if pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # W / 2 x H / 2 x 64

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU()
        if pool == 'avg':
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # W / 4 x H / 4 x 128

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU()
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_4 = nn.ReLU()
        if pool == 'avg':
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # W / 8 x H / 8 x 256

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU()
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_4 = nn.ReLU()
        if pool == 'avg':
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # W / 16 x H / 16 x 512

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU()
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_4 = nn.ReLU()

        if pool == 'avg':
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # W / 32 x H / 32 x 512