#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import os, psutil, subprocess
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset
from ummon.data import UnsupTensorDataset

__all__ = ["Timer"]

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[{}]".format(self.name))
        print("Elapsed: {:.4f} ms".format(time.time() - self.tstart))


def get_abs_avg_weights(model):
    weights = []
    for p in model.parameters():
        print(type(p))
        weights.append(p.data)
    return weights

def register_nan_checks_(model):
    def check_grad(module, input, output):
        if not hasattr(module, "weight"):
            return
        if any(np.all(np.isnan(gi.cpu().data.numpy())) for gi in module.weight if gi is not None):
            print('NaN weights in ' + type(module).__name__)
        if any(np.all(gi.cpu().data.numpy() > 1.) for gi in module.weight if gi is not None):
            print('Exploding weights in ' + type(module).__name__)
    handles = []
    for module in model.modules():
        handles.append(module.register_forward_hook(check_grad))
    return handles

def unregister_hooks_(handles):
    for handle in handles:
        handle.remove()
        
        
def get_shape_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    # CASE SUPERVISED
    if type(dataset[0]) == tuple:    
        # CASE MULTIPLE INPUT
        if type(dataset[0][0]) == tuple:
            data_shape = "["
            for di in dataset[0][0]:
                data_shape = data_shape + str(di.numpy().shape) + " "
            data_shape = data_shape + "]"
        else:
            data_shape = str(dataset[0][0].numpy().shape)
        # CASE MULTIPLE OUTPUT
        if type(dataset[0][1]) == tuple:
            target_shape = "["
            for di in dataset[0][1]:
                target_shape = target_shape + str(di.numpy().shape if isinstance(di, torch.Tensor)  else type(dataset[0][1])).__name__  + " "
            target_shape = target_shape + "]"
        else:
            target_shape = str(dataset[0][1].numpy().shape if isinstance(dataset[0][1], torch.Tensor) else type(dataset[0][1]).__name__ ) 
        return "Shape-IN:{}/TARGET:{}".format(data_shape,target_shape)
    # CASE UNSUPERVISED 
    else:
        # CASE MULTIPLE INPUT
        if type(dataset[0]) == tuple:
            data_shape = "["
            for di in dataset[0]:
                data_shape = data_shape + str(di.numpy().shape) + " "
            data_shape = data_shape + "]"
        else:
            data_shape = str(dataset[0].numpy().shape)
        return "Shape-IN:{}/".format(data_shape)

def get_type_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    # CASE SUPERVISED
    if type(dataset[0]) == tuple:  
        # CASE MULTIPLE INPUT
        if type(dataset[0][0]) == tuple:
            data_type = "["
            for di in dataset[0][0]:
                data_type = data_type + str(di.numpy().dtype) + " "
            data_type = data_type + "]"
        else:
            data_type = str(dataset[0][0].numpy().dtype)
        # CASE MULTIPLE OUTPUT
        if type(dataset[0][1]) == tuple:
            target_type = "["
            for di in dataset[0][1]:
                target_type = target_type + str(di.numpy().dtype if isinstance(di, torch.Tensor)  else type(dataset[0][1]).__name__)  + " "
            target_type = target_type + "]"
        else:
            target_type = str(dataset[0][1].numpy().dtype if isinstance(dataset[0][1], torch.Tensor) else type(dataset[0][1]).__name__)
        return "Type-IN:{}/TARGET:{}".format(data_type,target_type)
     # CASE UNSUPERVISED 
    else:
        # CASE MULTIPLE INPUT
        if type(dataset[0]) == tuple:
            data_type = "["
            for di in dataset[0]:
                data_type = data_type + str(di.numpy().dtype) + " "
            data_type = data_type + "]"
        else:
            data_type = str(dataset[0].numpy().dtype)
        return "Type-IN:{}/".format(data_type)


def get_size_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    
    return len(dataset) if dataset is not None else 0

def get_data_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    
    return (get_size_information(dataset), get_shape_information(dataset), get_type_information(dataset))

def check_precision(dataset, model, precision=None):
    if precision is None:
        precision = next(model.parameters()).cpu().data.numpy().dtype
    # CASE MULTIPLE INPUT
    if type(dataset[0][0]) == tuple:
        for di in dataset[0][0]:
            if not di.numpy().dtype == next(model.parameters()).cpu().data.numpy().dtype == precision:
                return False
        return True                
    else:
        # CASE UNSUPERVISED INPUT
        if type(dataset[0]) != tuple:
            return dataset[0].numpy().dtype == next(model.parameters()).cpu().data.numpy().dtype == precision
        # CASE SUPERVISED INPUT
        else:
            return dataset[0][0].numpy().dtype == next(model.parameters()).cpu().data.numpy().dtype == precision

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

def construct_dataset_from_tuple(logger, data_tuple, train = True):
    """
    Constructs a dataset from a given data tuple.
        
    Valid tuples are:
        (np.ndarray, np.ndarray, bs)
        (torch.tensor, torch.tensor, bs)
        (np.ndarray, bs)
        (torch.tensor, bs)
        np.ndarray
        torch.tensor
        
    Arguments:
        * logger     : ummon.Logger
        * data_tuple : input data
        * train      : specifies training or prediction mode where no labels are available (optional)
        
    Return:
        * dataset :  torch.utils.data.Datase

    """
    if train == True and len(data_tuple) != 3 and len(data_tuple) != 2:
            logger.error('Training data must be provided as a tuple (X,(y),batch) or as PyTorch DataLoader.',
                TypeError)
    if train == False and len(data_tuple) != 2 and type(data_tuple) != 1 and type(data_tuple) != np.ndarray and not istensor(data_tuple):
                logger.error('Validation data must be provided as a tuple (X,(y)) or as PyTorch DataLoader.',
                    TypeError)          
 
    # SUPERVISED
    if (len(data_tuple) == 2 and train == False) or (len(data_tuple) == 3 and train == True):
        # extract training data
        Xtr = data_tuple[0]
        ytr = data_tuple[1]
        if istensor(Xtr):
            Xtr = Xtr.numpy()
        if istensor(ytr):
            ytr = ytr.numpy()
        
        Xtrn, ytrn = check_data(logger, Xtr, ytr)
        
        # construct pytorch dataloader from 2-tupel
        x = torch.from_numpy(Xtrn)
        y = torch.from_numpy(ytrn)
        precision = Xtr.dtype
        if precision == np.float32:
            if ytr.dtype == np.int64:
                dataset = TensorDataset(x.float(), y.long())
            else:
                dataset = TensorDataset(x.float(), y.float())
        elif precision == np.float64:
            if ytr.dtype == np.int64:
                dataset = TensorDataset(x.double(), y.long())
            else:
                dataset = TensorDataset(x.double(), y.double())
        else:
            logger.error(str('Precision: ' + precision + ' is not supported yet.'))
        return dataset
    # UNSUPERVISED
    else:
        if type(data_tuple) != np.ndarray and not istensor(data_tuple):
            # extract training data
            Xtr = data_tuple[0]
        else:
            Xtr = data_tuple
        if istensor(Xtr):
            Xtr = Xtr.numpy()
        # construct pytorch dataloader from 2-tupel
        x = torch.from_numpy(Xtr)
        precision = Xtr.dtype
        if precision == np.float32:
            dataset = UnsupTensorDataset(x.float())
        elif precision == np.float64:
            dataset = UnsupTensorDataset(x.double())
        else:
            logger.error(str('Precision: ' + precision + ' is not supported yet.'))
        return dataset
        
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



def get_proc_memory_info():
    try:
        process = psutil.Process(os.getpid())
        percentage = process.memory_percent()
        memory = process.memory_info()[0] / float(2 ** 30)
        return {"mem" : memory,
              "usage" : percentage}
    except Exception:
        return 0.
  
def get_cuda_memory_info():
    """
    Get the current gpu usage.
    
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    try:
        if torch.cuda.is_available() == False:
            return 0.
        result = subprocess.check_output(
            [
                'nvidia-smi', '--query-gpu=memory.used',
                '--format=csv,nounits,noheader'
            ]).decode('utf-8')
        # Convert lines into a dictionary
        gpu_memory = [int(x) for x in result.strip().split('\n')]
        gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
        return gpu_memory_map
    except Exception:
        return 0.
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def tensor_tuple_to_variables(the_tuple):
    assert type(the_tuple) == tuple or type(the_tuple) == list
    assert isinstance(the_tuple[0], torch.Tensor)
    return tuple([Variable(t) for t in the_tuple])        
    
def tensor_tuple_to_cuda(the_tuple):
    assert type(the_tuple) == tuple or type(the_tuple) == list
    return tuple([t.cuda() for t in the_tuple])            


def online_average(value, count, avg):
    # BACKWARD COMPATIBILITY FOR TORCH < 0.4
    if type(value) is not float and not isinstance(value, np.float):
        if type(value) == torch.Tensor:
            value = value.item()
        else:
            value = value.data[0]
    navg = avg + (value - avg) / count
    return navg
     
def moving_average(t, ma, value, buffer):
    """
    Helper method for computing moving averages.
    
    Arguments
    ---------
    * t (int) : The timestep
    * ma (float) : Current moving average
    * value (float) : Current value
    * buffer (List<float>) : The buffer of size N
    
    Return
    ------
    * moving_average (float) : The new computed moving average.
    
    """
    # BACKWARD COMPATIBILITY FOR TORCH < 0.4
    if type(value) is not float and not isinstance(value, np.float):
        if type(value) == torch.Tensor:
            value = value.item()
        else:
            value = value.data[0]
    
    n = buffer.shape[0]
    if ma is None:
        moving_average = value
        buffer += value
    else:
        moving_average = ma + (value / n) - (buffer[t % n] / n)
    buffer[t % n] = value
    return moving_average
