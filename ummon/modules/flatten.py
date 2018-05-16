import numpy as np
import torch
import torch.nn as nn


__all__ = [ 'Flatten', 'Unflatten' ]


# Flatten
class Flatten(nn.Module):
    '''
    Class for flattening an input sequence::
    
        flat0 = Flatten([m,n,p])
    
    creates a flattening layer for an unflattened sequence of input 
    tensors. 
    
    This layer converts a sequence of multichannel images in a 4d tensor in a matrix 
    where each row is a flattened version of the multichannel image. The input 
    must be of size [mini_batch_size, nmaps, height, width]. The corresponding output 
    size is [1, 1, mini_batch_size, n] with rows of length n = nmaps * height * width. 
    The conversion of the unflattened to the flattened format is necessary for
    applying linear layers and loss functions. The conversion back to the unflattened 
    format is done in the layer type Unflatten.
    '''
    # constructor
    def __init__(self, insize):
        super(Flatten, self).__init__()
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[bs,{},{},{}]->[1,1,bs,{}])'.format(self.insize[0], self.insize[1], \
                self.insize[2], self.insize[0]*self.insize[1]*self.insize[2]) 
    
    
    def forward(self, input):
        '''
        Flattening of a 4D mini batch into a 2D output batch. The 
        leading dimension (training example index) is kept unchanged
        '''
        return input.view(-1, self.insize[0]*self.insize[1]*self.insize[2])
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block. Works only 
        when the output block is a single voxel or the entire layer. In both cases the
        input keeps the same format, but the single voxel gets a different position.
        '''
        # single unit
        if outp[2] == outp[5]:
            z0 = outp[2]//(self.insize[2] * self.insize[3])
            y0 = (outp[2] - z0 * self.insize[2] * self.insize[3]) // self.insize[3]
            x0 = outp[2] - z0 * self.insize[2] * self.insize[3] - y0 * self.insize[3]
            z1 = z0
            y1 = y0
            x1 = x0
        
        # full output
        else:
            z0 = 0
            y0 = 0
            x0 = 0
            z1 = self.insize[1] - 1
            y1 = self.insize[2] - 1
            x1 = self.insize[3] - 1
        
        return [z0, y0, x0, z1, y1, x1]


# Unflatten
class Unflatten(nn.Module):
    '''
    layer for unflattening an input sequence::
    
        unflat0 = Unflatten(id, [q], [m,n,p])
    
    creates an unflattening layer for a flattened sequence in a 
    matrix.
    
    This layer converts a sequence of flattened examples (given as rows of a matrix) 
    into a 4d tensor where each sequence element is a multichannel image. The input 
    must be of size [1, 1, mini_batch_size, q] with rows of length 
    q = m * n * p. The corresponding output size is [mini_batch_size, m, n, p]. The 
    conversion of the flattened to the unflattened format is 
    necessary for applying convolutional layers. The conversion back to the flattened 
    format is done in the layer type Flatten. 
    '''
    # constructor
    def __init__(self, insize, outsize):
        super(Unflatten, self).__init__()
        
        # check layer parameters
        if type(insize) == list: # allow both for list as input size (ummon style) or direct input
            if len(insize) != 1:
                raise TypeError('Input size list must have 1 element.')
            self.insize = int(insize[0])
        else:
            self.insize = int(insize)            
        self.outsize = list(outsize)
        if len(self.outsize) != 3:
            raise TypeError('Output size list must have 3 elements.')
        for s in self.outsize:
            s = int(s)
        if self.insize != self.outsize[0] * self.outsize[1] * self.outsize[2]:
            raise ValueError('Input size must be the product of the input sizes.')
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[1,1,bs,{}]->[bs,{},{},{}])'.format(self.insize, self.outsize[0], \
                self.outsize[1], self.outsize[2]) 
    
    # Forward path
    def forward(self, input, **kwargs):
        '''
        Unflattening of a 2D mini batch into a 4D output batch. The 
        leading dimension (training example index) is kept unchanged
        '''
        return input.view(-1, *self.outsize)
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block. Works only 
        when the output block is a single voxel or the entire layer. In both cases the
        input keeps the same format, but the single voxel gets a different position.
        '''
        if outp[2] == outp[5]:
            x0 = outp[0]*self.outsize[1]*self.outsize[2] + outp[1]*self.outsize[2] + outp[2]
            x1 = outp[3]*self.outsize[1]*self.outsize[2] + outp[4]*self.outsize[2] + outp[5]
            return [0, 0, x0, 0, 0, x1]
        else:
            return [0, 0, 0, 0, 0, self.insize-1]

