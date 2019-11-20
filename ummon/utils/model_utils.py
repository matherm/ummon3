import numpy as np
from torch.utils.data import DataLoader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def check_precision(dataset, model, precision=None):
    if isinstance(dataset, DataLoader):
        dataset = dataset.dataset
    if precision is None:
        precision = next(model.parameters()).to('cpu').data.numpy().dtype
    # CASE MULTIPLE INPUT
    if type(dataset[0][0]) == tuple or type(dataset[0][0]) == list:
        for di in dataset[0][0]:
            if not di.detach().numpy().dtype == next(model.parameters()).to('cpu').data.numpy().dtype == precision:
                return False
        return True                
    else:
        # CASE UNSUPERVISED INPUT
        if type(dataset[0]) != tuple and  type(dataset[0]) != list:
            return dataset[0].detach().numpy().dtype == next(model.parameters()).to('cpu').data.numpy().dtype == precision
        # CASE SUPERVISED INPUT
        else:
            return dataset[0][0].detach().numpy().dtype == next(model.parameters()).to('cpu').detach().numpy().dtype == precision

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
        if any(np.all(np.isnan(gi.to('cpu').data.numpy())) for gi in module.weight if gi is not None):
            print('NaN weights in ' + type(module).__name__)
        if any(np.all(gi.to('cpu').data.numpy() > 1.) for gi in module.weight if gi is not None):
            print('Exploding weights in ' + type(module).__name__)
    handles = []
    for module in model.modules():
        handles.append(module.register_forward_hook(check_grad))
    return handles

def unregister_hooks_(handles):
    for handle in handles:
        handle.remove()