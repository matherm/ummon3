import numpy as np
import torch
import torch.nn as nn

__all__ = [ 'LRN' ]

# local response normalization
class LRN(nn.LocalResponseNorm):
    '''
    Local response normalization layer::
    
        lnr0 = LNR([n,p,q], width, offset, alpha, beta)
    
    This layer provides a local response normalization for an input tensor of size 
    [:,n,p,q] with several feature maps according to Krizhevsky (2012). The feature maps 
    are averaged over a local neighborhood in the channel direction. Input and output 
    sizes are identical.
    
    Parameters:
    
    1. width: window size of neighborhood for averaging.
    2. offset: An offset (usually positive to avoid dividing by 0.
    3. alpha: A scale factor, usually positive.
    4. beta: Exponent of the denominator for normalization.
    
    '''
    # constructor
    def __init__(self, insize, width=9, offset=1.0, alpha=1.0, beta=0.5):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 3:
            raise TypeError('Input size list must have 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        width = int(width)
        if width < 3:
            raise ValueError('Normalization window must have a width >= 3.')
        if width > self.insize[0]:
            raise ValueError('Normalization window must have a width smaller than the number of feature maps.')
        if width % 2 == 0:
            raise ValueError('Normalization window size must be odd.')
        self._width = width
        offset = float(offset)
        if offset < 0:
            raise ValueError('Offset must be >= 0.')
        self._offset = offset
        alpha = float(alpha)
        beta = float(beta)
        if alpha <= 0.0 or beta <= 0.0:
            raise ValueError('Alpha and beta must be > 0.')
        self._alpha = alpha
        self._beta = beta
        
        super(LRN, self).__init__(self._width, self._alpha, self._beta, self._offset)
        
        self.outsize = self.insize
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[bs,{},{},{}]->[bs,{},{},{}];w={};k={:.2f};a={:.5f};be={:.2f})'.format(
            self.insize[0], self.insize[1], self.insize[2], self.outsize[0], self.outsize[1], 
            self.outsize[2], self._width, self._offset, self._alpha, self._beta)
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        z0 = outp[0] - self._width//2
        if z0 < 0:
            z0 = 0
        z1 = outp[3] + self._width//2
        if z1 >= self.insize[1]:
            z1 = self.insize[1] - 1
        
        return [z0, outp[1], outp[2], z1, outp[4], outp[5]]

