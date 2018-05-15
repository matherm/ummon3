#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3') 
sys.path.insert(0,'../ummon3')     
#############################################################################################

import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .logger import Logger
import ummon.utils as uu

__all__ = ["MetaAnalyzer"]

class MetaAnalyzer:
    """
    This class provides a generic analyzer for PyTorch-models. 
    For specialized models like Regression or Classification this class musst be subclassed.
    
    Methods
    -------
    compute_roc()       : [Not implemented yet]

    """
    
    def _evaluate(*args):
        raise NotImplementedError("This class is superclass")
    
    
    # generate an evaluation string used by logging module
    @staticmethod
    def evalstr(learningstate):
        raise NotImplementedError("This class is superclass")
    
    
    @staticmethod
    def _online_average(data, count, avg):
        # BACKWARD COMPATIBILITY FOR TORCH < 0.4
        if type(data) is not float:
            if type(data) == torch.Tensor:
                data = data.item()
            else:
                data = data.data[0]
        navg = avg + (data - avg) / count
        return navg
    
    
    @staticmethod
    def compute_roc(model, dataset):
        raise NotImplementedError
        pass
