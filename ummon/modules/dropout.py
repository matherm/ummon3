import numpy as np
import torch
import torch.nn as nn

__all__ = [ 'Dropout' ]

# Dropout layer class
class Dropout(nn.Dropout):
    '''
    Dropout layer class::
    
        dro0 = Dropout([n], rate)
        dro0 = Dropout([n,p], rate)
        dro0 = Dropout([n,p,q], rate)
    
    creates a dropout layer either for a flattened input sequence with input size 'n' or 
    for input of size [n,p] or [n,p,q]. 
    
    Applies dropout regularization to the input layer, i.e. during training a certain
    percentage of neurons (controlled by the dropout rate) are randomly set to zero.
    The output of the remaining neurons are scaled by the factor 1/(1 - dropout rate) to 
    maintain the overall activation of the layer. For every mini batch other neurons are 
    switched on and off individually. This forces the subsequent layer to develop robust 
    features and avoids overfitting. 
    Input and output sizes are identical. For prediction, the state of the network must
    set by calling model.eval(). For training mode (default), you have to call 
    model.train().
    '''
    # constructor
    def __init__(self, insize, rate, inplace=False):
        
        # check layer parameters
        self.insize = list(insize)
        if len(self.insize) not in [1,2,3]:
            raise TypeError('Input size list must have 1 to 3 elements.')
        for s in self.insize:
            s = int(s)
            if s < 1:
                raise ValueError('Input size must be > 0.')
        self.outsize = self.insize
        
        super(Dropout, self).__init__(rate, inplace)
        
        # network stats
        self.num_neurons = self.num_weights = self.num_adj_weights = 0
    
    
    # return printable representation
    def __repr__(self):
        insize = ''
        for s in self.insize:
            insize += ',{}'.format(s)
        return self.__class__.__name__ + '(' \
            + '[bs{}]->[bs{}];rate={})'.format(insize, insize, self.p)
    
    
    # Get the input block of a given output block
    def get_input_block(self, outp):
        '''
        Returns the input block that feeds into the specified output block.
        '''
        return outp  # inp = outp size

