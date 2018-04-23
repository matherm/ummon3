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
    inference()         : Computes model outputs
    compute_roc()       : [Not implemented yet]

    """
    
    def _evaluate(*args):
        raise NotImplementedError("This class is superclass")
    
    
    @staticmethod
    def _online_average(data, count, avg):
        navg = avg + (data - avg) / count
        return navg
    
    
    @staticmethod
    def inference(model, dataset, logger=Logger()):
        """
        Computes the output of a model for a given dataset
        
        Arguments
        ---------
        model           : nn.module
                          The model
        dataset         : torch.utils.data.Dataset OR tuple (X,y)
                          Dataset to evaluate
        logger          : ummon.Logger (Optional)
                          The logger to be used for output messages
        
        Return
        ------
        torch.tensor
        The output
        """
        # simple interface: training and test data given as numpy arrays
        if type(dataset) == tuple:
                 dataset = uu.construct_dataset_from_tuple(logger, dataset, train=False)
                 
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(model, nn.Module)
        assert uu.check_precision(dataset, model)
        
        use_cuda = next(model.parameters()).is_cuda
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, sampler=None, batch_sampler=None)  
        for i, data in enumerate(dataloader, 0):

                # Get the inputs
                inputs, targets = data
                
                 # Handle cuda
                if use_cuda:
                    inputs = inputs.cuda()
                
                # Execute Model
                model.eval()
                output = model(Variable(inputs))
                model.train()
        if type(output) != tuple:
            return output.cpu().data
        else:
            return output
    
    
    @staticmethod
    def compute_roc(model, dataset):
        raise NotImplementedError
        pass
