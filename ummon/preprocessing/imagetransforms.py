#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3') 
sys.path.insert(0,'../ummon3')     
#############################################################################################

import torch
import numpy as np

__all__ = [ 'VGG19Features']

def to_translated(image, final_size=(60,60)):
    '''
    This method pastes the given image into a greater image specified by final_size.
    
    Arguments
    --------
    Image as [Channels, Height, Width]
    
    Returns
    -------
    New Image as [Channels, (final_size)]
    '''
    assert type(image) == torch.FloatTensor
    assert image.max() <= 1
    assert image.numpy().ndim == 3

    greater_im = torch.FloatTensor(image.size(0), final_size[0], final_size[1]).zero_() 
    # COMPUTE RANDOM POSITION
    x = int(np.random.uniform(0,final_size[0] - image.size(1)))
    y = int(np.random.uniform(0,final_size[1] - image.size(2)))
    # PLACE IMAGE INTO NEW POSITION
    greater_im[:,x:x+image.size(1),y:y+image.size(2)] = image  
    return greater_im


def binarize(x, double = False):
    '''
    Binarize Image
    '''
    m = torch.distributions.Uniform(0, 1)
    xb = m.sample(x.size())
    bin_image = (x > xb).float() * 1
    if double == True:
        return bin_image.double()
    else:
        return bin_image


import torch.nn as nn
from ummon.modules.vgg19 import VGG19
from ummon.modules.gram import GramMatrix
from torchvision.models import vgg19

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
    def __init__(self, features = "pool4", gram = False, triangular=False, cachedir = None, cuda = False):
        """
        Parameters
        ----------
            *features (list or str):  Features: ['pool4', 'pool1', 'pool2', 'relu1_1', 'pool3', 'relu1_2']
            *gram (bool) : compute gram
            *triangular (bool) : triangular matrix 
            *cachedir (str) : an directory for caching computed matrices
            *cuda (bool) : compute with cuda
        
        """
        self.features = features
        self.cachedir = cachedir
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
            raise NotImplementedError
            
    def copy_weights_(self, model_source, model_dest):
        for i, k in enumerate(model_dest._modules):
            if isinstance(model_dest._modules[k], nn.Conv2d):
                model_dest._modules[k].weight = model_source[i].weight
                model_dest._modules[k].bias = model_source[i].bias
        

    def __call__(self, x):
        if x.dim != 3 and x.size(0) != 3:
            raise ValueError("Shape of input should be (3, x, x)")
        x = x.unsqueeze(0)
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
                    return y.cpu()[0]
                else:
                    result.append(y.view(-1))
            
        return torch.cat(result).cpu()

        
    