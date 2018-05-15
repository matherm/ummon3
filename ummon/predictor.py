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
import torch.nn.functional as F
from .logger import Logger
from .modules.loss import *
import ummon.utils as uu
import numpy as np

__all__ = ["Predictor"]

class Predictor:
    """
    This class provides a generic predictor for PyTorch-models. 
    For specialized models like Regression or Classification this class musst be subclassed.
    
    Methods
    -------
    predict()         : Computes model outputs
    """

    @staticmethod
    def predict(model, dataset, batch_size = -1, output_transform=None, logger=Logger()):
        """
        Computes the output of a model for a given dataset
        
        Arguments
        ---------
        model           : nn.module
                          The model
        dataset         : torch.utils.data.Dataset OR numpy X
                          Dataset to evaluate
        batch_size      : int
                          batch size used for evaluation (default: -1 == ALL)
        output_transform: nn.functionals
                          A functional that gets applied to the output. This can be useful when
                          combined loss like CrossEntropy was used during training.
        logger          : ummon.Logger (Optional)
                          The logger to be used for output messages
        
        Return
        ------
        torch.tensor
        The output
        """
        # simple interface: training and test data given as numpy arrays
        if type(dataset) == tuple or type(dataset) == np.ndarray or uu.istensor(dataset):
                 dataset = uu.construct_dataset_from_tuple(logger, dataset, train=False)
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(model, nn.Module)
        assert uu.check_precision(dataset, model)
        
        model.eval()
        use_cuda = next(model.parameters()).is_cuda
        bs = len(dataset) if batch_size == -1 else batch_size
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, sampler=None, batch_sampler=None)  
        outbuf = []
        for i, data in enumerate(dataloader, 0):

                # Get the inputs
                inputs = data
                
                 # Handle cuda
                if use_cuda:
                    inputs = inputs.cuda()
                
                # Execute Model
                output = model(Variable(inputs))
                
                # Apply output transforms
                if output_transform is not None:
                    if output_transform.__name__ == 'softmax':
                        output = output_transform(output, dim = 1)
                    else:
                        output = output_transform(output)
                # Save output for later evaluation
                outbuf.append(output.data.cpu())
                
        model.train()
        return torch.cat(outbuf, dim=0)
    
    
     # Get index of class with max probability
    @staticmethod
    def classify(output, loss_function = None):
        """
        Return
        ------
        classes (torch.LongTensor) - Shape [B x 1]
        """
        if loss_function is not None:
            # Evaluate non-linearity in case of a combined loss-function like CrossEntropy
            if isinstance(loss_function, torch.nn.BCEWithLogitsLoss):
                output = F.sigmoid(Variable(output)).data
            
            if isinstance(loss_function, torch.nn.CrossEntropyLoss):
                output = F.softmax(Variable(output), dim=1).data
        
        # Case single output neurons (e.g. one-class-svm sign(output))
        if (output.dim() > 1 and output.size(1) == 1) or output.dim() == 1:
            # makes zeroes positive
            classes = (output + 1e-12).sign().long()    
        
        # One-Hot-Encoding
        if (output.dim() > 1 and output.size(1) > 1):
            classes = output.max(1, keepdim=True)[1] 
        return classes
    
    
    @staticmethod
    def compute_accuracy(classes, targets):
        assert targets.shape[0] == classes.shape[0]
        
        # Case single output neurons (e.g. one-class-svm sign(output))
        if (targets.dim() > 1 and targets.size(1) == 1) or targets.dim() == 1:
            # Sanity check binary case
            if not targets.max() > 1:
                # Transform 0,1 encoding to -1 +1
                targets = (targets.float() - 1e-12).sign().long()
        
        # Classification one-hot coded targets are first converted in class labels
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = targets.max(1, keepdim=True)[1]
       
        if not isinstance(targets, torch.LongTensor):
            targets = targets.long()
        
        # number of correctly classified examples
        correct = classes.eq(targets.view_as(classes))
        
        # BACKWARD COMPATBILITY FOR TORCH < 0.4
        sum_correct = correct.sum()
        
        if type(sum_correct) == torch.Tensor:
            sum_correct = sum_correct.item()
        
        # accuracy
        accuracy = sum_correct / len(targets)
        return accuracy
    
    
    
    

