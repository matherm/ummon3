#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3') 
sys.path.insert(0,'../ummon3')     
#############################################################################################

import torch
import numpy as np
import torch.nn as nn
from ummon.modules.vgg19 import VGG19
from ummon.modules.gram import GramMatrix
from torchvision.models import vgg19
from pathlib import Path
import os

__all__ = ['FeatureCache']

def props(cls):   
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

def hash_obj(obj):
    hashes = []
    for key in props(obj):
        hashes.append(str(hash(key) + hash(str(getattr(obj, key)))))
    return str(hash("".join(hashes)))
    
class FeatureCache():
    """
    Simple feature cache.
    
    Usage
    ======
            transform = FeatureCache(VGG(features="pool4"), cachedir = None, clearcache = True)
            transform(tensor)
            
            OR
            
            vgg = FeatureCache(VGG19Features(features="pool4"), cachedir = None, clearcache = True)
            my_transforms = transforms.Compose([transforms.ToTensor(), vgg])
            test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
    
    Input
    ======
        *tensor (torch.Tensor) with shape (B x 3 x min 32 x 32)

    Return
    =====
        *feature (torch.Tensor)
    
    """
    def __init__(self, transform, cachedir="__ummoncache__", clearcache = True):
        """
        Parameters
        ----------
            *transform (torchvison.transform) : the transformation to cache    
            *cachedir (str) : an directory for caching computed matrices
            *clearcache (bool) : deletes cache on object construction
            
        """
        self.transform = transform
        self.cachedir = cachedir
        self.clearcache = clearcache
        self.unique_transformation_hash = hash_obj(self.transform)
        
        self.cache_hits = 0
        self.cache_misses = 0
        
        # create cache dir
        if os.path.exists(self.cachedir) == False:
            os.makedirs(self.cachedir)

        if clearcache == True:
            [os.remove(os.path.join(self.cachedir, f)) for f in os.listdir(self.cachedir) if f.endswith(".npy")]
    
    
    def __call__(self, x):
        is_numpy = not isinstance(x, torch.Tensor)         
        is_cuda = x.is_cuda  if not is_numpy else False
        
        # Handle cache lookup
        if is_numpy:
            input_raw_data = x.data.tobytes()
        else:
            input_raw_data = x.detach().cpu().numpy().data.tobytes()
        fname = "".join(self.unique_transformation_hash) + "_" + str(hash(input_raw_data))
        path = str(self.cachedir + "/ummon_" + fname + ".npy")
        
        if Path(path).exists():
            self.cache_hits += 1    # stats
            cached_raw_data = np.load(path)
            if is_numpy:
                return cached_raw_data
            elif is_cuda:
                return torch.from_numpy(cached_raw_data).cuda()
            else:
                return torch.from_numpy(cached_raw_data)
        else:
            self.cache_misses += 1 # stats
            
                
        # Compute transformation
        y = self.transform(x)
        
        ## Handle cache put
        np.save(path, y.cpu().detach().numpy())
                
        return y
        

         
         
        