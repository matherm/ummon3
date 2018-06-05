import numpy as np
import torch
import torch.nn as nn

__all__ = [ 'Dropout' ]

# Dropout layer class
class Dropout(nn.Dropout):
    '''
    Dropout layer class::
    
        dro0 = Dropout(n, rate)
    
    creates a dropout layer for a flattened input sequence with input size 'n'. 
    
    Applies dropout regularization to the input layer, i.e. during training a certain
    percentage of neurons (controlled by the dropout rate) are randomly set to zero.
    The output of the remaining neurons are scaled by the factor 1/(1 - dropout rate) to 
    maintain the overall activation of the layer. For every mini batch other neurons are 
    switched on and off individually. This forces the subsequent layer to develop robust 
    features and avoids overfitting. Currently, dropout is only implemented for fully 
    connected layers, i.e. the input must be in flattened format [1, 1, mini_batch_size, n].
    Input and output sizes are identical. For prediction, the state of the network must
    set by calling model.eval(). For training mode (default), you have to call 
    model.train().
    '''
    # constructor
    def __init__(self, insize, rate, inplace=False):
        
        # check layer parameters
        if type(insize) == list:
            insize = insize[0]
        insize = int(insize)
        if insize < 1:
            raise ValueError('Input size must be > 0.')
        self.insize = [1, 1, insize]
        self.outsize = self.insize
        
        super(Dropout, self).__init__(rate, inplace)
        
        # network stats
        self.num_neurons = self.num_weights = self.num_adj_weights = 0
    
    
    # return printable representation
    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + '[1,1,bs,{}]->[1,1,bs,{}];rate={})'.format(self.insize[2], self.outsize[2], 
            self.p)
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        return outp  # inp = outp size

