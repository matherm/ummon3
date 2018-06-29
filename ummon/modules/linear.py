import numpy as np
import torch
import torch.nn as nn


__all__ = [ 'Linear' ]


class Linear(nn.Linear):
    '''
    Fully connected linear layer class::
    
        line0 = Linear([n], num_neurons, init, bias)
    
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
    The layer accepts only flat input tensors of size [1,1,mbs,n] and produces output tensors 
    of size [1,1,mbs,n]. If the input is in unflattened format, a Flatten node has to be 
    included as input layer. If the argument 'bias' is set to False the bias vector is 
    ignored.
    
    Attributes:
    
    * w: weight matrix
    * b: bias vector
    
    '''    
    def __init__(self, insize, num_neurons=10, init='xavier_normal', bias=True):
        
        # allow both for list as input size (ummon style) and for number of input neurons
        if type(insize) == list:
            insize = int(insize[0])
        
        # instantiate pyTorch linear layer
        super(Linear, self).__init__(insize, num_neurons, bias)
        
        # initialize weights
        init = str(init)
        nn.init.__dict__[init](self.weight)
        self.insize = [1,1,self.in_features]
        self.outsize = [1,1,self.out_features]
        self.num_weights = self.in_features*self.out_features
        if self.bias is not None:
            nn.init.constant_(self.bias,0.0)
            self.num_weights += self.out_features
        self.num_adj_weights = self.num_weights
        self.num_neurons = self.out_features
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[1,1,bs,{}]->[1,1,bs,{}])'.format(str(self.in_features), 
            str(self.out_features)) + ', bias=' + str(self.bias is not None) + ')' 
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block. Since this is
        a fully connected layer, the entire input layer is returned as a block.
        '''
        return [0, 0, 0, 0, 0, self.in_features - 1]
    
    
    # get weights
    @property
    def w(self):
        return self.weight.data.numpy()
    
    
    # set weights
    @w.setter
    def w(self, wmat):
        if type(wmat) != np.ndarray:
            raise TypeError('Provided weight matrix is not a *NumPy* array')
        if wmat.shape != (self.out_features, self.in_features):
            raise TypeError('Provided weight tensor has wrong size.')
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
            if type(bvec) != np.ndarray:
                raise TypeError('Provided bias vector is not a *NumPy* array')
            if bvec.ndim == 1:
                bvec = bvec.reshape((len(bvec),1)).copy()
            if bvec.shape != (self.out_features, 1):
                raise TypeError('Provided bias vector has wrong size.')
            self.bias.data = torch.from_numpy(bvec)

