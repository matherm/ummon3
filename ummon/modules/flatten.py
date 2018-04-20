import numpy as np
import torch
import torch.nn as nn


__all__ = [ 'Flatten' ]


# Flatten
class Flatten(nn.Module):
    '''
    Class for flattening an input sequence::
    
        flat0 = Flatten(id, [m,n,p], cnet, ['name 1', 'name 2',..])
    
    creates a flattening layer with ID string 'id' for an unflattened sequence of input 
    tensors. The layer is registered in the network 'cnet' and connected to the input 
    layers 'name 1', 'name 2 ' etc.
    
    This node converts a sequence of multichannel images in a 4d tensor in a matrix 
    where each row is a flattened version of the multichannel image. The input 
    must be of size [mini_batch_size, nmaps, height, width]. The corresponding output 
    size is [1, 1, mini_batch_size, n] with rows of length n = nmaps * height* width. 
    The conversion of the unflattened to the flattened format is necessary for
    applying linear layers and loss nodes. The conversion back to the unflattened 
    format is done in the node type Unflatten.
    '''
    # constructor
    def __init__(self, insize):
        super(Flatten, self).__init__()
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        print(self.insize)
    
    
    # return printable representation
    def __repr__(self):
        print(self.insize[0]*self.insize[1]*self.insize[2])
        return self.__class__.__name__ + '(' \
            + '[1,{},{},{}]->[1,1,1,{}])'.format(self.insize[0], self.insize[1], \
                self.insize[2], self.insize[0]*self.insize[1]*self.insize[2]) 
    
    def forward(self, input):
        '''
        Flattening of a 4D mini batch into a 2D output batch. The 
        leading dimension (training example index) is kept unchanged
        '''
        return input.view(-1, self.insize[0]*self.insize[1]*self.insize[2])
