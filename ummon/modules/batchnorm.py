import numpy as np
import torch
import torch.nn as nn

__all__ = [ 'BatchNorm1d' ]

# 1 d batch normalization
class BatchNorm1d(nn.BatchNorm1d):
    '''
    1 d batch normalization::
    
        bano = BatchNorm1d([n,p], eps, momentum, affine, track_running_stats)
    
    creates a 1d batch normalization layer which often improves convergence and learning
    speed. For the theory, see Ioffe, S. and Szegedy, C. (2015). Batch normalization: 
    Accelerating deep network training by reducing internal covariate shift. arXiv preprint 
    arXiv:1502.03167. For the meaning of the parameters, see PyTorch documenation of
    nn.BatchNorm1d. The input tensor for this layer must be in a non-flattened 1d format,
    each example consisting of 'p' features and 'n' channels. For examples, see the test 
    in ummon_test.py.
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

