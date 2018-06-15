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
    
    Supported shapes are ( 3, y, x) or (y, x).
    Please notice that normalization should happe nafter anomaly transformation because otherwise we end up with grey values.
    
    """
    def __init__(self, size=16, color=0, propability=0.2):
         self.anom_size = size
         self.propability = propability
    
    def __call__(self, x):
        assert np.max(x.numpy()) <= 1 and np.min(x.numpy()) >= 0
        assert x.size(0) < x.size(1) and x.size(0) < x.size(2)
        if np.random.rand() < self.propability:
            if x.dim() == 3:
                _y = np.random.randint(0, x.size(1) - self.anom_size)
                _x = np.random.randint(0, x.size(2) - self.anom_size)
                x[:, _y : _y + self.anom_size, _x : _x + self.anom_size] = 0
            else:
                _y = np.random.randint(0, x.size(0) - self.anom_size)
                _x = np.random.randint(0, x.size(1) - self.anom_size)
                x[_y : _y + self.anom_size, _x : _x + self.anom_size] = 0
            return x
        else:
            return x
            