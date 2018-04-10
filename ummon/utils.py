#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import numpy as np
import time
import torch

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
            
def unregister_hooks(handles):
    for handle in handles:
        handle.remove()

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
            target_shape = target_shape + str(di.numpy().shape if type(di) != int else "int")  + " "
        target_shape = target_shape + "]"
    else:
        target_shape = str(dataset[0][1].numpy().shape if type(dataset[0][1]) != int else "int") 
    return "Shape-IN:{}/OUT:{}".format(data_shape,target_shape)

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
            target_type = target_type + str(di.numpy().dtype if type(di) != int else "int")  + " "
        target_type = target_type + "]"
    else:
        target_type = str(dataset[0][1].numpy().dtype if type(dataset[0][1]) != int else "int")
    return "Type-IN:{}/OUT:{}".format(data_type,target_type)

def get_size_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    
    return len(dataset) if dataset is not None else 0

def get_data_information(dataset):
    if dataset is None: return "---"
    assert isinstance(dataset, torch.utils.data.Dataset)
    
    return (get_size_information(dataset), get_shape_information(dataset), get_type_information(dataset))