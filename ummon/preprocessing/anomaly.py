#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 09:13:39 2018

@author: Matthias Hermann
"""
import numpy as np
import torch

__all__ = ['SquareAnomaly', 'GaussianNoiseAnomaly', 'LaplacianNoiseAnomaly', 'LineDefectAnomaly', 'TurtleAnomaly', 'ShuffleAnomaly']

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
            x = torch.from_numpy(np.asarray(x)).float()
        else:
            was_numpy = False
        x = x.clone()
            
        assert np.min(x.numpy()) >= 0
        if x.dim() == 2: x = x.unsqueeze(2)
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
            
        _y = np.random.randint(0, x.size(0) - self.anom_size)
        _x = np.random.randint(0, x.size(1) - self.anom_size)
        x[_y : _y + self.anom_size, _x : _x + self.anom_size, :] = self.color
            
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
        my_transforms = transforms.Compose([GaussianNoiseAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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
            x = torch.from_numpy(np.asarray(x)).float()
        else:
            was_numpy = False
        x = x.clone()
        assert np.min(x.numpy()) >= 0
        if x.dim() == 2: x = x.unsqueeze(2)
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        
        size = self.anom_size if self.anom_size != -1 else x.shape[0]
            
        _y = np.random.randint(0, x.size(0) - size) if size < x.size(0) else 0
        _x = np.random.randint(0, x.size(1) - size) if size < x.size(1) else 0
        
        noise = torch.from_numpy(np.random.normal(self.mean, self.std, (size, size, x.size(2))).astype(np.float32))
        x[_y : _y + size, _x : _x + size, :] += noise
        x = torch.clamp(x, 0, 1)
            
        if was_numpy:
            return x.numpy()
        else:
            return x

class LaplacianNoiseAnomaly():
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
        my_transforms = transforms.Compose([LaplacianNoiseAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, size=-1, loc=0, scale=1):
         self.anom_size = size
         self.loc = loc
         self.scale = scale
    
    def __call__(self, x):
        if not torch.is_tensor(x):
            was_numpy = True
            x = torch.from_numpy(np.asarray(x)).float()
        else:
            was_numpy = False
        x = x.clone()
        assert np.min(x.numpy()) >= 0
        if x.dim() == 2: x = x.unsqueeze(2)
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        
        size = self.anom_size if self.anom_size != -1 else x.shape[0]
            
        _y = np.random.randint(0, x.size(0) - size) if size < x.size(0) else 0
        _x = np.random.randint(0, x.size(1) - size) if size < x.size(1) else 0
        
        noise = torch.from_numpy(np.random.laplace(self.loc, self.scale, (size, size, x.size(2))).astype(np.float32))
        x[_y : _y + size, _x : _x + size, :] += noise
        x = torch.clamp(x, 0, 1)
            
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
        Please notice that normalization should happen after anomaly transformation because otherwise we end up with grey values.
    
    Usage
    =====
        my_transforms = transforms.Compose([LineDefectAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, width=1, intensity=0,  channel=[0], additive=False, vertical=True):
         self.anom_size = width
         self.intensity = intensity
         self.additive = additive
         self.vertical = vertical
         if type(channel) == list:
             self.channel = channel
         else:
             self.channel = [channel]

    def __call__(self, x, pos=-1):
        if not torch.is_tensor(x):
            was_numpy = True
            x = torch.from_numpy(np.asarray(x)).float()
        else:
            was_numpy = False
        x = x.clone()
            
        assert np.min(x.numpy()) >= 0
        if x.dim() == 2: x = x.unsqueeze(2)
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        
        # Handle different scaling
        x = x.float()
        if x.max() > 1:
            amax = 255
            defect = self.intensity
            if defect <= 1 or defect >= -1:
                defect = defect * 255
        else:
            amax = 1.
            defect = self.intensity
            if defect > 1 or defect < -1:
                defect = defect / 255

        if pos == -1:
            _x = np.random.randint(0, x.size(1) - self.anom_size)
        elif 0 <= pos <= (x.size(1) - self.anom_size ):
            _x = pos
        else:
            raise ValueError(pos + ' invalid value.')

        if self.additive:
            for c in self.channel:
                if self.vertical:
                    x[:, _x : _x + self.anom_size, c] += defect
                else:
                    x[_x : _x + self.anom_size, : , c] += defect
                x = torch.clamp(x, 0, amax)
        else:
            for c in self.channel:
                if self.vertical:
                    x[:, _x : _x + self.anom_size, c] = abs(defect)
                else:
                    x[_x : _x + self.anom_size, : , c] = abs(defect)

        if was_numpy:
            return x.numpy()
        else:
            return x
        
        
        
class TurtleAnomaly():
    """
    Adds a TurtleAnomaly to a given patch.
    A turtle moves across the patch and places for every random choosen direction a color from the bucket.
    
    Limitations
    ===========
        Supported shapes are ( y, x, 3) or (y, x) or ( y, x, 1).
    
    Note
    =====
        Please notice that normalization should happe nafter anomaly transformation because otherwise we end up with grey values.
    
    Usage
    =====
        my_transforms = transforms.Compose([TurtleAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, pixels=4, color_bucket=[0], thickness=1, channel = -1):
        # SIZE
        self.pixels = pixels
        self.thickness = thickness
        # CHANNEL 
        if channel == -1:
            self.channel = [0,1,2]
        else:
            if type(channel) == list:
                self.channel = channel
            else:
                self.channel = [channel]
         #COLOR
        if type(color_bucket) == list:
            self.color_bucket = color_bucket
        else:
            self.color_bucket = [color_bucket]
    
    def __call__(self, x):
        if not torch.is_tensor(x):
            was_numpy = True
            x = torch.from_numpy(np.asarray(x)).float()
        else:
            was_numpy = False
        x = x.clone()
        x = x.float()
            
        assert np.min(x.numpy()) >= 0
        if x.dim() == 2: x = x.unsqueeze(2)
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        
        x_pos = np.random.randint(0,x.size(1))
        y_pos = np.random.randint(0,x.size(0))
        
        for step in range(self.pixels):
            defect = self.color_bucket[np.random.randint(0,len(self.color_bucket))]
            # Handle different scaling
            if x.max() > 1:
                defect = defect * 255 if defect <= 1 else defect
            else:
                defect = defect / 255 if defect > 1 else defect
            
            for c in self.channel:
                x[y_pos:y_pos+self.thickness, x_pos:x_pos+self.thickness, c] = defect
               
            # Update the turtle in one of four directions
            while True:
                move = np.random.randint(0,4)
                x_pos_new, y_pos_new = np.inf, np.inf
                if move == 0:
                    x_pos_new = x_pos + 1 if x_pos < x.size(1) - self.thickness else np.inf
                    y_pos_new = y_pos
                if move == 1:
                    x_pos_new = x_pos - 1 if x_pos > 0 + self.thickness - 1  else np.inf
                    y_pos_new = y_pos
                if move == 2:
                    y_pos_new = y_pos + 1 if y_pos < x.size(0) - self.thickness else np.inf
                    x_pos_new = x_pos
                if move == 3:
                    y_pos_new = y_pos - 1 if y_pos > 0 + self.thickness - 1  else np.inf
                    x_pos_new = x_pos
                # Break out
                if  x_pos_new  != np.inf and y_pos_new != np.inf:
                    x_pos, y_pos =  x_pos_new, y_pos_new 
                    break
            
                
                
        if was_numpy:
            return x.numpy()
        else:
            return x
        
        
class ShuffleAnomaly():
    """
    Shuffles a a patch randomly.
    
    Limitations
    ===========
        Supported shapes are ( y, x, 3) or (y, x) or ( y, x, 1).
    
    
    Usage
    =====
        my_transforms = transforms.Compose([ShuffleAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, mode="rowscols"):
         assert mode in ["rowscols", "full"]
         if mode == "rowscols":
             self.full_shuffle = False
         else:
             self.full_shuffle = True
         
    def __call__(self, x):
        if not torch.is_tensor(x):
            was_numpy = True
            x = torch.from_numpy(np.asarray(x)).float()
        else:
            was_numpy = False
        x = x.clone()
        assert np.min(x.numpy()) >= 0
        if x.dim() == 2: x = x.unsqueeze(2)
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        
        x = x.numpy()
        if self.full_shuffle:
            np.random.shuffle(x.reshape(-1))
        else:
            if x.ndim == 3:
                np.random.shuffle(x.reshape(-1,x.shape[2]))
            else:
                np.random.shuffle(x.reshape(-1))
            
        if was_numpy:
            return x
        else:
            return torch.from_numpy(x)


class RotationAnomaly():
    """
    Rotatesa patch by 90 or 180 degree.
    
    Limitations
    ===========
        Supported shapes are ( y, x, 3) or (y, x) or ( y, x, 1).
    
    
    Usage
    =====
        my_transforms = transforms.Compose([RotationAnomaly(rot=90), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    """
    def __init__(self, rot=90):
         assert rot in [90, 180, 270]
         self.rot = rot

    def __call__(self, x):
        if not torch.is_tensor(x):
            was_numpy = True
            x = torch.from_numpy(np.asarray(x)).float()
        else:
            was_numpy = False
        x = x.clone()
        if x.dim() == 2: 
            x = x.unsqueeze(2)
        assert np.min(x.numpy()) >= 0
        assert x.size(2) < x.size(1) and x.size(2) < x.size(0)
        
        x = x.numpy()
        x = np.rot90(x, self.rot // 90).copy()
      #  from scipy.ndimage.interpolation import affine_transform
      #  x = affine_transform(x, np.array([[np.cos(25),-np.sin(25),0],
      #                                     [np.sin(25),np.cos(25),0],
      #                                       [0,0,1]]))       
        if was_numpy:
            return x
        else:
            return torch.from_numpy(x)
