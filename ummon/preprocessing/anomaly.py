#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:13:39 2018

@author: Fabian Freiberg
"""

import numpy as np
import torch
class SquareAnomaly():
    """
    Adds a random square anomaly to a given patch.
    
    Limitations
    ===========
        Supported shapes are ( y, x, 3) or (y, x) or ( y, x, 1).
    
    Note
    =====
        Please notice that normalization should happe nafter anomaly transformation because otherwise we end up with grey values.
    
    Usage
    =====
        my_transforms = transforms.Compose([SquareAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, size=16, color=0):
         self.anom_size = size
         self.color = color
    
    def __call__(self, x):
        if not torch.is_tensor(x):
            was_numpy = True
            x = torch.from_numpy(x)
        else:
            was_numpy = False
        x = x.clone()
            
        assert np.min(x.numpy()) >= 0
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        if x.dtype == torch.float and x.max() > 1.:
            # Data is float and we have a range greater [0, 255]
            if self.color <= 1:
                # color is defined in range [0, 1]
                self.color = self.color * 255
        
        if x.dtype == torch.float and x.min() >= 0 and x.max() <= 1.:
            # Data is float and we have a range between [0, 1]
            if self.color > 1:
                # color is defined in range [0, 255]
                self.color = self.color / 255
            
        if x.dim() == 3:
            _y = np.random.randint(0, x.size(0) - self.anom_size)
            _x = np.random.randint(0, x.size(1) - self.anom_size)
            x[_y : _y + self.anom_size, _x : _x + self.anom_size, :] = self.color
        else:
            _y = np.random.randint(0, x.size(0) - self.anom_size)
            _x = np.random.randint(0, x.size(1) - self.anom_size)
            x[_y : _y + self.anom_size, _x : _x + self.anom_size] = self.color
            
        if was_numpy:
            return x.numpy()
        else:
            return x
        
        
class GaussianNoiseAnomaly():
    """
    Adds a random noise to a given patch.
    
    Limitations
    ===========
        Supported shapes are ( y, x, 3) or (y, x) or ( y, x, 1).
    
    Note
    =====
        Please notice that normalization should happe nafter anomaly transformation because otherwise we end up with grey values.
    
    Usage
    =====
        my_transforms = transforms.Compose([NoiseAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, size=-1, mean=0, std=1):
         self.anom_size = size
         self.mean = mean
         self.std = std
    
    def __call__(self, x):
        if not torch.is_tensor(x):
            was_numpy = True
            x = torch.from_numpy(x)
        else:
            was_numpy = False
            
        assert np.min(x.numpy()) >= 0
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        
        size = self.anom_size if self.anom_size != -1 else x.shape[0]
            
        _y = np.random.randint(0, x.size(0) - size) if size < x.size(0) else 0
        _x = np.random.randint(0, x.size(1) - size) if size < x.size(1) else 0
        if x.dim() == 3:
            noise = torch.from_numpy(np.random.normal(self.mean, self.std, (size, size, 3)).astype(np.float32))
            x[_y : _y + size, _x : _x + size, :] += noise
        else:
            noise = torch.from_numpy(np.random.normal(self.mean, self.std, (size, size)).astype(np.float32))
            x[_y : _y + self.anom_size, _x : _x + size] += noise
            
        if was_numpy:
            return x.numpy()
        else:
            return x
        
        
class LineDefectAnomaly():
    """
    Adds a line defect tto a given patch.
    
    Limitations
    ===========
        Supported shapes are ( y, x, 3) or (y, x) or ( y, x, 1).
    
    Note
    =====
        Please notice that normalization should happe nafter anomaly transformation because otherwise we end up with grey values.
    
    Usage
    =====
        my_transforms = transforms.Compose([LineDefectAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, width=1, channel=[0]):
         self.anom_size = width
         if type(channel) == list:
             self.channel = channel
         else:
             self.channel = [channel]
    
    def __call__(self, x):
        if not torch.is_tensor(x):
            was_numpy = True
            x = torch.from_numpy(x)
        else:
            was_numpy = False
        x = x.clone()
            
        assert np.min(x.numpy()) >= 0
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        
        # Handle different scaling
        x = x.float()
        if x.max() > 1:
            defect = 255.
        else:
            defect = 1
        
        if x.dim() == 3:
            _x = np.random.randint(0, x.size(0) - self.anom_size)
            for c in self.channel:
                x[:, _x + self.anom_size, c] = defect
        else:
           x[:, _x + self.anom_size] = defect
            
        if was_numpy:
            return x.numpy()
        else:
            return x