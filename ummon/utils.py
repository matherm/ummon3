#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data.dataset import TensorDataset
import os, psutil, subprocess


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print("[{}]".format(self.name))
        print("Elapsed: {:.4f} ms".format(time.time() - self.tstart))

class Torchutils:
    def __init__(self, name=None):
        self.name = "ummon.Torchutils"

    @staticmethod        
    def get_abs_avg_weights(model):
        weights = []
        for p in model.parameters():
            print(type(p))
            weights.append(p.data)
        return weights
    
    @staticmethod
    def register_nan_checks(model):
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
    
    @staticmethod            
    def unregister_hooks(handles):
        for handle in handles:
            handle.remove()
            
            
    @staticmethod
    def get_shape_information(dataset):
        if dataset is None: return "---"
        assert isinstance(dataset, torch.utils.data.Dataset)
        
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
    
    @staticmethod
    def get_type_information(dataset):
        if dataset is None: return "---"
        assert isinstance(dataset, torch.utils.data.Dataset)
        
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
    
    @staticmethod
    def get_size_information(dataset):
        if dataset is None: return "---"
        assert isinstance(dataset, torch.utils.data.Dataset)
        
        return len(dataset) if dataset is not None else 0
    
    @staticmethod
    def get_data_information(dataset):
        if dataset is None: return "---"
        assert isinstance(dataset, torch.utils.data.Dataset)
        
        return (Torchutils.get_size_information(dataset), Torchutils.get_shape_information(dataset), Torchutils.get_type_information(dataset))
    
    @staticmethod
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
            return dataset[0][0].numpy().dtype == next(model.parameters()).cpu().data.numpy().dtype == precision
    
    
    # check input data
    @staticmethod
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
            if y.dtype != 'float32':
                #TODO: This causes trouble as some losses (e.g. crossentropy expect long-types
                y = y.astype('float32')
        return X, y
    
    @staticmethod
    def construct_dataset_from_tuple(logger, data_tuple, train = True):
        if train == True and len(data_tuple) != 3:
                logger.error('Training data must be provided as a tuple (X,y,batch) or as PyTorch DataLoader.',
                    TypeError)
        if train == False and len(data_tuple) != 2:
                    logger.error('Validation data must be provided as a tuple (X,y) or as PyTorch DataLoader.',
                        TypeError)          
       
        # extract training data
        Xtr = data_tuple[0]
        ytr = data_tuple[1]
        Xtrn, ytrn = Torchutils.check_data(logger, Xtr, ytr)
        
        # construct pytorch dataloader from 2-tupel
        x = torch.from_numpy(Xtrn)
        y = torch.from_numpy(ytrn)
        precision = Xtr.dtype
        if precision == np.float32:
            if ytr.dtype == np.int64:
                dataset = TensorDataset(x.float(), y.long())
            else:
                dataset = TensorDataset(x.float(), y.float())
        if precision == np.float64:
            if ytr.dtype == np.int64:
                dataset = TensorDataset(x.double(), y.long())
            else:
                dataset = TensorDataset(x.double(), y.double())
        return dataset


    @staticmethod
    def get_proc_memory_info():
        try:
            process = psutil.Process(os.getpid())
            percentage = process.memory_percent()
            memory = process.memory_info()[0] / float(2 ** 30)
            return {"mem" : memory,
                  "usage" : percentage}
        except Exception:
            return 0.
      
    @staticmethod    
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
        
    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    @staticmethod
    def transform_model(model, precision, use_cuda = False):
        assert isinstance(model, nn.Module)
        assert precision == np.float32 or precision == np.float64
        
        # Computational configuration
        if precision == np.float32:
            model = model.float()
        if precision == np.float64:
            model = model.double()
        if precision == np.int32:
            # TODO: Custom model conversion FPGA-Teamproject
            pass
        if use_cuda:
            assert torch.cuda.is_available() == True
            model = model.cuda()
        else:
            model = model.cpu()
        return model
            
        