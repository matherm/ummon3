#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:44:12 2018

Originally taken from: https://github.com/leongatys/PytorchNeuralStyleTransfer

@author: Fabian Freiberg
"""
import torch.nn as nn
import torch

__all__ = [ 'GramMatrix']

class GramMatrix(nn.Module):
    def forward(self, x):
        b, c, h, w = x.size()
        F = x.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h*w)
        return G
