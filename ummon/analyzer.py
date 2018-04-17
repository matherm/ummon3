#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import time
import numpy as np
import torch
import torch.nn as nn
from ummon.trainingstate import Trainingstate
from ummon.logger import Logger
from ummon.utils import Torchutils
from torch.utils.data import DataLoader
from torch.autograd import Variable

class Analyzer:
    """
    This class provides a generic analyzer for PyTorch-models. For a given PyTorch-model it 
    computes statistical information about the model, e.g. accuracy, loss, ROC, etc.
    
    
    Methods
    -------
    evaluate()          : Evaluates a model with given validation dataset
    classify()          : Classifies model outputs with One-Hot-Encoding            
    inference()         : Computes model outputs
    compute_accuracy()  : Computes the accuracy of a classification
    compute_roc()       : [Not implemented yet]
             
    """
    def __init__(self):
        self.name = "ummon.Analyzer"
            
            
    @staticmethod    
    def evaluate(model, loss_function, dataset, regression, logger=Logger(), after_eval_hook=None, batch_size=-1):
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
        regression      : bool
                          Specifies if a classification needs to be done
        logger          : ummon.Logger (Optional)
                          The logger to be used for output messages
        after_eval_hook : OPTIONAL function(model, output.data, targets.data, loss.data)
                          A hook that gets called after forward pass
        batch_size      : int
                          batch size used for evaluation (default: -1 == ALL)
        
        Return
        ------
        Dictionary
        A dictionary containing keys `loss`, `accuracy`, ´samples_per_second`, `detailed_loss`, 'args[]`
        """
        # simple interface: training and test data given as numpy arrays
        if type(dataset) == tuple:
                 dataset = Torchutils.construct_dataset_from_tuple(logger, dataset, train=False)
        
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(model, nn.Module)
        assert Torchutils.check_precision(dataset, model)
        
        use_cuda = next(model.parameters()).is_cuda
        evaluation_dict = {}
        loss_average, acc_average = 0.,0.
        bs = len(dataset) if batch_size == -1 else batch_size
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, sampler=None, batch_sampler=None)
        output_buffer = []
        for i, data in enumerate(dataloader, 0):
                
                # Take time
                t = time.time()

                # Get the inputs
                inputs, targets = data
                
                # Handle cuda
                if use_cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()
                
                # Execute Model
              #  model.eval()
                output = model(Variable(inputs))
               # model.train()
                
                # Compute Loss
                targets = Variable(targets)
                loss = loss_function(output, targets).cpu()
                loss_average = Analyzer._online_average(loss.data[0], i + 1, loss_average)
                
                # Run hook
                if after_eval_hook is not None:
                    if type(output) != tuple:
                        after_eval_hook(model, output.data, targets.data, loss.data)
                    else:
                        after_eval_hook(model, output[0].data, targets.data, loss.data)
                
                 # Save output for later evaluation
                if type(output) != tuple:
                    output_buffer.append((output.data.clone(), targets.data.clone(), i))
                else:
                    output_buffer.append((output[0].data.clone(), targets.data.clone(), i))
                
        # Compute classification accuracy
        if not regression:
            for saved_output, saved_targets, batch in output_buffer:
                if type(saved_output) != tuple:
                    classes = Analyzer.classify(saved_output.cpu())
                else:
                    classes = Analyzer.classify(saved_output[0].cpu())
                acc = Analyzer.compute_accuracy(classes, saved_targets.cpu())
                acc_average = Analyzer._online_average(acc, batch + 1, acc_average)
            evaluation_dict["accuracy"] = acc_average
        else:
            evaluation_dict["accuracy"] = 0.
        evaluation_dict["samples_per_second"] = dataloader.batch_size / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = repr(loss_function)
        evaluation_dict["args[]"] = {}
        del output_buffer[:]

        return evaluation_dict
    
    @staticmethod
    def _online_average(data, count, avg):
        navg = avg + (data - avg) / count
        return navg
    
    # Get index of class with max probability
    @staticmethod
    def classify(output):
        """
        Return
        ------
        classes (torch.LongTensor) - Shape [B x 1]
        """
        assert isinstance(output, torch.Tensor)
        classes = output.max(1, keepdim=True)[1] 
        return classes
    
    
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
                 dataset = Torchutils.construct_dataset_from_tuple(logger, dataset, train=False)
                 
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(model, nn.Module)
        assert Torchutils.check_precision(dataset, model)
        
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
    def compute_accuracy(classes, targets):
        assert isinstance(classes, torch.LongTensor)
        assert targets.shape[0] == classes.shape[0]
        
        # Classification one-hot coded targets are first converted in class labels
        if targets.dim() > 1:
            targets = targets.max(1, keepdim=True)[1]
        if not isinstance(targets, torch.LongTensor):
            targets = targets.long()
                
        # number of correctly classified examples
        correct = classes.eq(targets.view_as(classes))
        
        # accuracy
        accuracy = correct.sum() / len(targets)
        return accuracy
    
    
    @staticmethod
    def compute_roc(model, dataset):
        raise NotImplementedError
        pass
     
        
if __name__ == "__main__":
    pass
