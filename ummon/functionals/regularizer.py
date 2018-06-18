#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small module for collection often needed regularizers.

@author: matthias
"""

import torch

def l2_reg(params):
    l2 = torch.FloatTensor(1)
    for w in params:
        l2 += (w**2).sum()
        return l2  