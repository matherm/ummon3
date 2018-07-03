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
    
    @staticmethod    
    def evaluate(model, loss_function, dataset, logger=Logger(), after_eval_hook=None, batch_size=-1,
        output_buffer=None):
        """
        Evaluates a model with given validation dataset
        
        Arguments
        ---------
        model           : nn.module
                          The model
        loss_function   : nn.module
                          The loss function to evaluate
        dataset         : torch.utils.data.Dataset OR tuple (X,y)
                          Dataset to evaluate
        logger          : ummon.Logger (Optional)
                          The logger to be used for output messages
        after_eval_hook : OPTIONAL function(ctx, model, output.data, targets.data, loss.data)
                          A hook that gets called after forward pass
        batch_size      : int
                          batch size used for evaluation (default: -1 == ALL)
        
        Return
        ------
        Dictionary
        A dictionary containing keys `loss`, `accuracy`, ´samples_per_second`, `detailed_loss`, 'args[]`
        """
        # Input validation
        dataloader = SupervisedAnalyzer._data_validation(dataset, batch_size, logger)
        assert uu.check_precision(dataloader.dataset, model)
        
        use_cuda = next(model.parameters()).is_cuda
        evaluation_dict = {}
        ctx = {}
        loss_average = 0.

        # Take time
        t = time.time()
        
        for i, data in enumerate(dataloader, 0):
                

                # Get the inputs
                inputs, targets = data
                
                # Handle cuda
                if use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                # Execute Model
                output = model(Variable(inputs))
                
                # Compute Loss
                targets = Variable(targets)
                loss = loss_function(output, targets).cpu()
               
                loss_average = MetaAnalyzer._online_average(loss, i + 1, loss_average)
                
                # Run hook
                if after_eval_hook is not None:
                    ctx = after_eval_hook(ctx, model, output.data, targets.data, loss.data)
                
                
        evaluation_dict["training_accuracy"] = 0.0        
        evaluation_dict["accuracy"] = 0.0
        evaluation_dict["samples_per_second"] = len(dataloader) / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = repr(loss_function)
        
        return evaluation_dict
    
    
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
