import torch
from torch.autograd import Variable
import numpy as np

def istensor(t):
    return type(t) ==  torch.Tensor \
            or type(t) ==  torch.FloatTensor      \
            or type(t) ==  torch.DoubleTensor \
            or type(t) ==  torch.HalfTensor   \
            or type(t) ==  torch.ByteTensor   \
            or type(t) ==  torch.CharTensor  \
            or type(t) ==  torch.ShortTensor \
            or type(t) ==  torch.IntTensor   \
            or type(t) ==  torch.LongTensor  \

def tensor_tuple_to_variables(the_tuple):
    # for compatibility as transforming Tensor to Variable is not needed anymore
    return the_tuple     
    
def tensor_tuple_to_cuda(the_tuple):
    return tuple_to(the_tuple, "cuda")     

def tensor_tuple_to_cpu(the_tuple):
    return tuple_to(the_tuple, "cpu")     

def tensor_tuple_to_data(the_tuple):
    if type(the_tuple) != tuple and type(the_tuple) != list:
        return the_tuple.detach().to('cpu').data
    else:
        return tuple([t.detach().to('cpu').data for t in the_tuple])

def tuple_to(tuple_like, device):
    if type(tuple_like) != tuple and type(tuple_like) != list:
        return tuple_like.to(device)
    else:
        return tuple([t.to(device) for t in tuple_like])


def tuple_detach(tuple_like):
    if type(tuple_like) != tuple and type(tuple_like) != list:
        return tuple_like.detach()
    else:
        return tuple([t.detach() for t in tuple_like])


def check_data(logger, X, y=[]):
    '''
    Internal function for checking the validity and size compatibility of the provided 
    data.
    
    Arguments:
    
    * logger: ummon.Logger
    * X     : input data
    * y     : output data (optional)
    
    '''
    # check inputs
    if type(X) != np.ndarray:
        logger.error('Input data is not a *NumPy* array')
    if X.ndim > 5 or X.ndim == 0:
        logger.error('Input dimension must be 1..5.', TypeError)
    if X.dtype != 'float32':
        X = X.astype('float32')
    
    # convert into standard shape
    if X.ndim == 1:
        X = X.reshape((1, len(X))).copy()
    elif X.ndim == 3:
        X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2])).copy()
    elif X.ndim == 4:
        X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])).copy()
    
    # check targets
    if len(y) > 0:
        if type(y) != np.ndarray:
            logger.error('Target data is not a *NumPy* array')
        if y.ndim > 2 or y.ndim == 0:
            logger.error('Targets must be given as vector or matrix.')
        if y.ndim == 1:
            pass
            #TODO: This causes trouble as some losses (e.g. crossentropy expect a vector of size N)
            #y = y.reshape((len(y), 1)).copy()
        if np.shape(y)[0] != np.shape(X)[0]:
            logger.error('Number of targets must match number of inputs.')
    return X, y

