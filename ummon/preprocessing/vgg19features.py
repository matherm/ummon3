#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3') 
sys.path.insert(0,'../ummon3')     
#############################################################################################

import torch
import numpy as np
import torch.nn as nn
from ummon.modules.vgg19 import VGG19
from ummon.modules.gram import GramMatrix
from torchvision.models import vgg19
from pathlib import Path
import os

__all__ = ['VGG19Features']

class VGG19Features():
    """
    Extracts features from VGG19
    
    Usage
    ======
            transform = VGG(features="pool4")
            transform(tensor)
    
    Input
    ======
        *tensor (torch.Tensor) with shape (B x 3 x min 32 x 32)

    Return
    =====
        *feature (torch.Tensor)
    
    """
    def __init__(self, features = "pool4", gram = False, triangular=False, cachedir = None, clearcache = True, cuda = False):
        """
        Parameters
        ----------
            *features (list or str):  Features: ['pool4', 'pool1', 'pool2', 'relu1_1', 'pool3', 'relu1_2']
            *gram (bool) : compute gram
            *triangular (bool) : triangular matrix 
            *cachedir (str) : an directory for caching computed matrices
            *clearcache (bool) : deletes cache on object construction
            *cuda (bool) : compute with cuda
        
        """
        self.features = features
        self.cachedir = cachedir
        self.clearcache = clearcache
        self.gram = gram
        self.cuda = cuda
        self.triangular = triangular
        
        # Original PyTorch VGG19
        self.model = vgg19(pretrained=True).features
        
        # Custom VGG19 with better Layer naming convention
        self.vgg19 = VGG19()
        
        # Copy weights
        self.copy_weights_(self.model, self.vgg19)
        
        # Cuda
        self.vgg19 =  self.vgg19.cuda() if self.cuda and torch.cuda.is_available() else self.vgg19
        
        if type(self.features) is not list:
            self.features  = [self.features]
        
        if cachedir is None:
             self.cache = False
        else:
            self.cache = True

        if clearcache == True:
            [os.remove(f) for f in os.listdir(self.cachedir) if f.endswith(".npy")]
            
            
    def copy_weights_(self, model_source, model_dest):
        for i, k in enumerate(model_dest._modules):
            if isinstance(model_dest._modules[k], nn.Conv2d):
                model_dest._modules[k].weight = model_source[i].weight
                model_dest._modules[k].bias = model_source[i].bias
        

    def __call__(self, x):
        if x.dim != 3 and x.size(0) != 3:
            raise ValueError("Shape of input should be (3, x, x)")
        x = x.unsqueeze(0)
        
        if self.cache:
            fname = str(hash(str(x.detach().cpu().numpy())))
            path = str(self.cachedir + "/" + fname + ".npy")
            if Path(path).exists():
                return torch.from_numpy(np.load(path))
            
        result = []
        for name, layer in self.vgg19._modules.items():
            x = layer(x)
            y = x
            if self.gram:
                y = GramMatrix()(y)
                if self.triangular:
                    y = y[:, np.triu_indices(x.shape[1])[0], np.triu_indices(x.shape[1])[1]]
            if name in self.features:
                if len(self.features) == 1:
                    result = y.cpu()[0]
                else:
                    result.append(y.view(-1))
        
        if len(self.features) == 1:
            if self.cache:
                np.save(path, result.detach().numpy())
            return result
        else:    
            if self.cache:
                np.save(path, torch.cat(result).cpu().detach().numpy())
            return torch.cat(result).cpu()