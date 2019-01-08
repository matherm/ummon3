import numpy as np
import torch
import torch.nn as nn

__all__ = [ 'BatchNorm1d' ]

# 1 d batch normalization
class BatchNorm1d(nn.BatchNorm1d):
    '''
    1 d batch normalization::
    
        bn10 = BatchNorm1d([n,p,q], kernel_size, stride, padding)
    
    creates a max pooling layer. This method selects the local maximum in the 
    neighborhood determined by 'kernel_size' as the output value. The error is sparsely 
    backpropagated, i.e., only the pixels where the maximum occurred is updated with the 
    backpropagated error, all others are ignored.
    
    Applies max pooling to subsample an input tensor and provides a certain translation 
    invariance. The input tensor for this layer must be in a non-flattened format. Pooling is 
    controlled by 3 attributes: the stride between window centers in x- and y-direction
    ('stride': either a number or 2-tuple), and the pooling window size ('kernel_size': 
    either a number or 2-tuple). You can set an additional padding region ('padding':
    either a number or 2-tuple) filled with zeroes for treating the image boundary 
    regions.
    
    You obtain the classical pooling in the valid region (as, e.g., in LeNet) by setting
    stride and kernel size to the same values and by setting 'padding' to zero. If you
    want a behaviour similar to a strided convolution with zero padding set 'padding'
    to half the filter size in each direction. For examples, see the tests in ummon_test.py.
    '''    
    # constructor
    def __init__(self, insize, eps=1e-05, momentum=0.1, affine=True, 
        track_running_stats=True):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) != 2:
            raise TypeError('Input size list must have 2 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        
        super(BatchNorm1d, self).__init__(self.insize[0], eps, momentum, affine, 
            track_running_stats)
        
        # output size
        self.outsize = self.insize
        
        # network stats
        self.num_neurons = self.num_weights = self.num_adj_weights = 0
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[bs,{},{}]->[bs,{},{}];eps={};mom={};affine={};track={})'.format(self.insize[0], 
            self.insize[1], self.outsize[0], self.outsize[1], self.eps, self.momentum, 
            self.affine, self.track_running_stats)
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        return outp  # inp = outp size

