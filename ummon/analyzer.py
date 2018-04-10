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
    def evaluate(model, loss_function, dataset, regression, after_eval_hook=None):
        """
        Evaluates a model with given validation dataset
        
        Arguments
        ---------
        model           : nn.module
                          The model
        loss_function   : nn.module
                          The loss function to evaluate
        dataset         : torch.utils.data.Dataset
                          Dataset to evaluate
        regression      : bool
                          Specifies if a classification needs to be done
        after_eval_hook : OPTIONAL function(model, output.data, targets.data, loss.data)
                          A hook that gets called after forward pass
        
        Return
        ------
        Dictionary
        A dictionary containing keys `loss`, `accuracy`, ´samples_per_second`
        """
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(model, nn.Module)
        
        evaluation_dict = {}
        use_cuda = next(model.parameters()).is_cuda
        dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, sampler=None, batch_sampler=None)  
        for i, data in enumerate(dataloader, 0):
            
                # Take time
                t = time.time()

                # Get the inputs
                inputs, targets = data
                
                # Handle cuda
                if use_cuda:
                    inputs = inputs.cuda()
                
                # Execute Model
                model.eval()
                output = model(Variable(inputs)).cpu()
                model.train()
                
                # Compute Loss
                targets = Variable(targets)
                loss = loss_function(output, targets).cpu()
                
                # Run hook
                if after_eval_hook is not None:
                    after_eval_hook(model, output.data, targets.data, loss.data)
                
                # Compute classification accuracy
                if not regression:
                    classes = Analyzer.classify(output.data)
                    acc = Analyzer.compute_accuracy(classes, targets.data)
                    evaluation_dict["accuracy"] = acc
                else:
                    evaluation_dict["accuracy"] = 0.
                evaluation_dict["samples_per_second"] = dataloader.batch_size / (time.time() - t)
                evaluation_dict["loss"] = loss.data[0]
                evaluation_dict["detailed_loss"] = repr(loss_function)
                evaluation_dict["args[]"] = {}
                
        return evaluation_dict
        
    # Get index of class with max probability
    @staticmethod
    def classify(output):
        assert isinstance(output, torch.Tensor)
        classes = output.max(1, keepdim=True)[1] 
        return classes
    
    
    @staticmethod
    def inference(model, dataset):
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(model, nn.Module)
        
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
                output = model(Variable(inputs)).cpu()
                model.train()
        return output.cpu()
    
    
    @staticmethod
    def compute_accuracy(classes, targets):
        assert isinstance(classes, torch.LongTensor)
        assert targets.shape[0] == classes.shape[0]
        
        if not isinstance(targets, torch.LongTensor):
            targets = targets.long()
        
        # one-hot coded targets are first converted in class labels
        if targets.dim() > 1:
            targets = targets.max(1, keepdim=True)[1]
        
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
