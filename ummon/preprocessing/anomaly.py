#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:13:39 2018

@author: Fabian Freiberg
"""

import numpy as np
class SquareAnomaly():
    """
    Adds a random square anomaly to a given patch.
    
    Limitations
    ===========
        Supported shapes are ( 3, y, x) or (y, x).
    
    Note
    =====
        Please notice that normalization should happe nafter anomaly transformation because otherwise we end up with grey values.
    
    Usage
    =====
        my_transforms = transforms.Compose([transforms.ToTensor(), SquareAnomaly(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, size=16, color=0):
         self.anom_size = size
    
    def __call__(self, x):
        assert np.max(x.numpy()) <= 1 and np.min(x.numpy()) >= 0
        assert x.size(0) < x.size(1) and x.size(0) < x.size(2)
        if x.dim() == 3:
            _y = np.random.randint(0, x.size(1) - self.anom_size)
            _x = np.random.randint(0, x.size(2) - self.anom_size)
            x[:, _y : _y + self.anom_size, _x : _x + self.anom_size] = 0
        else:
            _y = np.random.randint(0, x.size(0) - self.anom_size)
            _x = np.random.randint(0, x.size(1) - self.anom_size)
            x[_y : _y + self.anom_size, _x : _x + self.anom_size] = 0
        return x
            