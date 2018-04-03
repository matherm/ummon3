from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


__all__ = [ 'Sequential' ]


class Sequential(nn.Sequential):
    '''
    Neural network with a single path. Same as torch.nn.Sequential, but with a simplified 
    interface. Usage::
    
        cnet = Sequential(
            ('line0', Linear([5], 7, 'xavier_uniform', 0.001)),
            ('sigm0', nn.Sigmoid()),
            ...
        )
    
    The layers are given as 2-tupels consisting of the layer name and decalaration.
    '''
    def __init__(self, *args):
        
        # get layers
        layers = list(args)
        
        # init nn.Sequential
        super(Sequential, self).__init__(OrderedDict(layers))
        
    
    # return printable representation
    def __repr__(self):
        tmpstr = self.__class__.__name__ + ':\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            tmpstr = tmpstr + '  ' + key + ': ' + modstr + '\n'
        return tmpstr
