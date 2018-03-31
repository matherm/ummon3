import numpy as np
import torch
import torch.nn as nn


__all__ = [ 'Linear' ]


class Linear(nn.Linear):
    '''
    Fully connected linear layer class::
    
        line0 = Linear(n, num_neurons, init, wdecay, bias)
    
    creates a fully connected linear layer with 'n' input neurons and
    'num_neurons' output neurons. The weight matrix is initialized
    according to 'init' which must be one of the initialization methods described in
    pyTorch. 
    
    The output of a layer with m neurons is computed as 
    
    .. math::
        Y = W X^T + b
    
    where the input is given as an mbs x n matrix X with the flattened n-dimensional
    inputs as rows and the mini batch size mbs, the output as an mbs x m matrix Y with 
    neuron outputs in a row, the m x n weight matrix W and the m x 1 bias vector b. 
    The node accepts only input tensors of size [1,1,mbs,n] and produces output tensors 
    of size [1,1,mbs,n]. If the input is in unflattened format, a Flatten node has to be 
    included as input layer. You can set an L2 weight decay 'wdecay' for the weights for 
    this layer individually.
    
    Attributes:
    
    * w: weight matrix
    * b: bias vector
    
    '''    
    def __init__(self, insize, num_neurons=10, init='xavier_normal', wdecay=0.0, bias=True):
        
        # allow both for list as input size (ummon style) and for number of input neurons
        if type(insize) == list:
            insize = int(insize[0])
        
        # instantiate pyTorch linear layer
        super(Linear, self).__init__(insize, num_neurons, bias)
        
        # initialize weights
        init = str(init)
        nn.init.__dict__[init](self.weight)
        if self.bias is not None:
            nn.init.constant(self.bias,0.0)
        
        # weight decay is only set at the beginning
        self._wdecay = float(wdecay)
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[1,1,1,{}]->[1,1,1,{}],wdec={})'.format(str(self.in_features), 
            str(self.out_features), str(self._wdecay)) \
            + ', bias=' + str(self.bias is not None) + ')' 
    
    # get weights
    @property
    def w(self):
        return self.weight.data.numpy()
    
    # set weights
    @w.setter
    def w(self, wmat):
        self.weight.data = torch.from_numpy(wmat)
    
    # get bias
    @property
    def b(self):
        if self.bias is not None:
            return self.bias.data.numpy()
        else:
            return None
    
    # set bias
    @b.setter
    def b(self, bvec):
        if self.bias is not None:
            self.bias.data = torch.from_numpy(bvec)
