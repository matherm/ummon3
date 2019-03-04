import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import ConcatDataset
from ummon.datasets import *
from ummon.logger import Logger

from .data_utils import istensor, check_data

def gen_dataloader(dataset, batch_size=-1, has_labels=True, logger=Logger()):
    """
    Does input data validation for training and validation data.
    
    Arguments
    ---------
    *dataset (torch.utils.data.Dataloader OR
                torch.utils.data.Dataset OR 
                numpy (X, y, (bs)) OR 
                torch.Tensor (X, y, (bs)) : A data structure holding the validation data.
    *batch_size (int) : The batch size
    *has_labels (bool) : Indicates whether tuple must have labels
    *logger (ummon.logger) : the logger 
    
    Return
    ------
    *dataloader (torch.utils.data.Dataloader) : Same as input or corrected versions from input.
    """
    # simple interface: training and test data given as numpy arrays
    if type(dataset) == tuple:
        data = dataset
        if len(dataset) != 2 and len(dataset) != 3:
            logger.error('Training data must be provided as a tuple (X, y, (bs)) or as PyTorch DataLoader.',TypeError)
        elif has_labels == True and len(dataset) == 3:
            #case (X, y, bs)
            batch_size = data[2]
            new_tuple = (data[0],data[1])
            torch_dataset = construct_dataset_from_tuple(new_tuple, logger=logger)
        elif has_labels == True and len(dataset) == 2:
            #case (X, y)
            new_tuple = (data[0], data[1])
            torch_dataset = construct_dataset_from_tuple(new_tuple, logger=logger)
        elif has_labels == False and len(dataset) == 2:
            #case (X, bs)
            batch_size = data[1]
            new_tuple = (data[0],)
            torch_dataset = construct_dataset_from_tuple(new_tuple, logger=logger)
        elif has_labels == False and len(dataset) == 1:
            #case (X)
            new_tuple = (data[0],)
            torch_dataset = construct_dataset_from_tuple(new_tuple, logger=logger)
        else:
            logger.error('Training data must be provided as a tuple (X, y, (bs)) or as PyTorch DataLoader.',TypeError)
            

    if isinstance(dataset, np.ndarray) or istensor(dataset):
        torch_dataset = construct_dataset_from_tuple(dataset, logger=logger)

    if isinstance(dataset, torch.utils.data.Dataset):
        torch_dataset = dataset
    
    if isinstance(dataset, torch.utils.data.dataset.Dataset):
        torch_dataset = dataset

    if isinstance(dataset, torch.utils.data.dataloader.DataLoader):
        return dataset


    if type(dataset) == list:
        dataloader = [dataset]
    else:
        bs = len(torch_dataset) if batch_size == -1 else batch_size
        dataloader = DataLoader(torch_dataset, batch_size=bs, shuffle=False, sampler=None, batch_sampler=None)        

    return dataloader

def construct_dataset_from_tuple(data_tuple, logger=Logger()):
    """
    Constructs a dataset from a given data tuple.
        
    Valid tuples are:
        (np.ndarray, np.ndarray)
        (torch.tensor, torch.tensor)
        (np.ndarray)
        (torch.tensor)
        np.ndarray
        torch.tensor
        
    Arguments:
        * logger     : ummon.Logger
        * data_tuple : input data
        * has_labels (bool) : Indicates whether tuple must have labels (Optional)
        
    Return:
        * dataset :  torch.utils.data.Datase
    
    """
    if isinstance(data_tuple, torch.utils.data.Dataset):
        return data_tuple     

    dataset = None
       
    # SUPERVISED
    if type(data_tuple) == tuple and len(data_tuple) == 2:
        Xtr = data_tuple[0]
        ytr = data_tuple[1]
        if istensor(Xtr):
            Xtr = Xtr.detach().numpy()
        if istensor(ytr):
            ytr = ytr.detach().numpy()
        
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
    
    # UNSUPERVISED
    if (type(data_tuple) == tuple and len(data_tuple) == 1) or type(data_tuple) == np.ndarray or istensor(data_tuple):
        if type(data_tuple) == tuple:
            Xtr = data_tuple[0]
        Xtr = data_tuple
        if istensor(Xtr):
            Xtr = Xtr.detach().numpy()
        x = torch.from_numpy(Xtr)
        precision = Xtr.dtype
        if precision == np.float32:
            dataset = UnsupTensorDataset(x.float())
        elif precision == np.float64:
            dataset = UnsupTensorDataset(x.double())
        else:
            logger.error(str('Precision: ' + precision + ' is not supported yet.'))
    
    if dataset is None:
        raise ValueError("Could not create dataset from tuple. Tuple was: " + str(type(data_tuple)))

    return dataset

def get_numerical_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    # GET A SAMPLE
    samples = []
    labels = []
    for i in range(np.min([len(dataset), 10])):
        # e.g. dataset = [([array, array],label),([array,array],label),..]
        # e.g. dataset = [((array, array),label),((array,array),label),..]
        # e.g. *dataset = [((array, array),(label, label)),((array,array),(label, label)),..]
        # e.g. dataset = [(array, label),(array,label),..]
        # e.g. dataset = [array, array, ...]
        datapoint = dataset[i]
        if type(datapoint) == tuple:
            X, y = datapoint   # X e (tensor, tensor) , y e (label, label)
        else:
            X = datapoint
            y = 0
        samples = samples + [x.numpy() for x in list(X) if istensor(x)]
        labels  = labels  + list(np.asarray(y).reshape(1,-1))
    if len(samples) > 0:
        data    =  "min:"  + str(np.round(np.min(samples),1)) \
                + " max:"  + str(np.round(np.max(samples),1)) \
                + " mean:" + str(np.round(np.mean(samples),1)) \
                + " std:"  + str(np.round(np.std(samples),1))

        labels  =  "min:"  + str(np.round(np.min(labels),1)) \
                + " max:"  + str(np.round(np.max(labels),1)) \
                + " mean:" + str(np.round(np.mean(labels),1)) \
                + " std:"  + str(np.round(np.std(labels),1)) 
        return "\n\tStats Data:{} / Labels:{}".format(data,labels)
    else:
        return "\n\tStats Dataset:{}".format(repr(dataset))

def get_type_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    # CASE SUPERVISED
    if type(dataset[0]) == tuple:  
        # CASE MULTIPLE INPUT
        if type(dataset[0][0]) == tuple:
            data_type = "["
            for di in dataset[0][0]:
                data_type = data_type + str(di.detach().numpy().dtype) + " "
            data_type = data_type + "]"
        else:
            data_type = str(dataset[0][0].detach().numpy().dtype)
        # CASE MULTIPLE OUTPUT
        if type(dataset[0][1]) == tuple:
            target_type = "["
            for di in dataset[0][1]:
                target_type = target_type + str(di.detach().numpy().dtype if torch.is_tensor(di)  else type(di).__name__)  + ", "
            target_type = target_type + "]"
        else:
            target_type = str(dataset[0][1].detach().numpy().dtype if torch.is_tensor(dataset[0][1]) else type(dataset[0][1]).__name__)
        return "\n\tType  IN:{} / TARGET:{}".format(data_type,target_type)
     # CASE UNSUPERVISED 
    else:
        if istensor(dataset[0]):
            data_type = str(dataset[0].detach().numpy().shape)
        else:
            data_type = repr(dataset[0])
        return "\n\tType  IN:{}".format(data_type)

def get_shape_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    # CASE SUPERVISED
    if type(dataset[0]) == tuple:    
        # CASE MULTIPLE INPUT
        if type(dataset[0][0]) == tuple:
            data_shape = "["
            for di in dataset[0][0]:
                data_shape = data_shape + str(di.detach().numpy().shape) + " "
            data_shape = data_shape + "]"
        else:
            data_shape = str(dataset[0][0].detach().numpy().shape)
        # CASE MULTIPLE OUTPUT
        if type(dataset[0][1]) == tuple:
            target_shape = "["
            for di in dataset[0][1]:
                target_shape = target_shape + str(di.detach().numpy().shape if torch.is_tensor(di)  else type(di).__name__)  + ", "
            target_shape = target_shape + "]"
        else:
            target_shape = str(dataset[0][1].detach().numpy().shape if torch.is_tensor(dataset[0][1]) else type(dataset[0][1]).__name__ ) 
        return "\n\tShape IN:{} / TARGET:{}".format(data_shape,target_shape)
    # CASE UNSUPERVISED 
    else:
        if istensor(dataset[0]):
            data_shape = str(dataset[0].detach().numpy().shape)
        else:
            data_shape = repr(dataset[0])
        return "\n\tShape IN:{}".format(data_shape)

def get_size_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    
    return len(dataset) if dataset is not None else 0

def get_data_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    
    return (get_size_information(dataset), get_shape_information(dataset), get_type_information(dataset), get_numerical_information(dataset))


def add_dataset_to_loader_(dataloader, merge_dataset):
    dataloader.dataset = ConcatDataset([dataloader.dataset, merge_dataset])
