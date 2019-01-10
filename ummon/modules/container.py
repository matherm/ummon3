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
        
        # check for duplicates
        keys = set()
        for key,_ in layers:
            if key in keys:
                raise TypeError('Duplicate layer name {} in network.'.format(key))
            keys.add(key)
        
        # init nn.Sequential
        super(Sequential, self).__init__(OrderedDict(layers))
        
    
    # return printable representation
    def __repr__(self):
        tmpstr = self.__class__.__name__ + ':\n'
        for key, module in self._modules.items():
            modstr = module.__repr__()
            tmpstr = tmpstr + '  ' + key + ': ' + modstr + '\n'
        tmpstr = tmpstr + '  -\n{0:25}: {1}\n'.format('  Number of layers', self.num_layers) + \
            '{0:25}: {1}\n'.format('  Number of neurons', self.num_neurons) + \
            '{0:25}: {1}\n'.format('  Number of weights', self.num_weights) + \
            '{0:25}: {1}\n'.format('  Number of adj. weights', self.num_adj_weights)
        return tmpstr
    
    # read only attribute: number of layers on path
    @property
    def num_layers(self):
        return len(self._modules.items())
    
    # read only attribute: number of neurons in network
    @property
    def num_neurons(self):
        n = 0
        for key, module in self._modules.items():
            if hasattr(module, 'num_neurons'):
                n += module.num_neurons
        return n
    
    # read only attribute: number of weights (independently from being shared or not)
    @property
    def num_weights(self):
        n = 0
        for key, module in self._modules.items():
            if hasattr(module, 'num_weights'):
                n += module.num_weights
        return n
    
    # read only attribute: number of adjustable weights 
    @property
    def num_adj_weights(self):
        n = 0
        for key, module in self._modules.items():
            if hasattr(module, 'num_adj_weights'):
                n += module.num_adj_weights
        return n
