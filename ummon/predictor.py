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
        dataset         : torch.utils.data.Dataset OR numpy X or torch.Tensor or torch.utils.data.DataLoader
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
        dataloader = uu.gen_dataloader(dataset, has_labels=False, batch_size=batch_size, logger=logger)
        assert isinstance(model, nn.Module)
        assert uu.check_precision(dataloader, model)
        
        use_cuda = next(model.parameters()).is_cuda
        device = "cuda" if use_cuda else "cpu"
        
        model.eval()
        outbuf = []
        for i, data in enumerate(dataloader, 0):
            
                # Get the inputs
                inputs = uu.input_of(data)
                
                # Handle cuda
                inputs = uu.tuple_to(inputs, device)
                output = uu.tuple_detach(model(inputs))
                
                # Apply output transforms
                if output_transform is not None:
                    if output_transform.__name__ == 'softmax':
                        output = output_transform(output, dim = 1)
                    else:
                        output = output_transform(output)
                # Save output for later evaluation
                outbuf.append(output.data)
                
        model.train()
        full_output = torch.cat(outbuf, dim=0).to('cpu')
        if type(dataset) == np.ndarray:
            return full_output.numpy()
        else:
            return full_output

    @staticmethod
    def predict_loss(model, loss, dataset, batch_size = -1, output_transform=None, logger=Logger()):
        """
        Computes the output of a model for a given dataset
        
        Arguments
        ---------
        model           : nn.module
                          The model
        loss            : nn.Module
                          The loss function as func(x, y)
        dataset         : torch.utils.data.Dataset OR numpy X or torch.Tensor or torch.utils.data.DataLoader
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
        The losses (N, 1)
        """
        # simple interface: training and test data given as numpy arrays
        dataloader = uu.gen_dataloader(dataset, has_labels=True, batch_size=batch_size, logger=logger)
        assert isinstance(model, nn.Module)
        assert uu.check_precision(dataloader, model)
        
        use_cuda = next(model.parameters()).is_cuda
        device = "cuda" if use_cuda else "cpu"
        
        model.eval()
        outbuf = []
        for i, data in enumerate(dataloader, 0):
            
                # Get the inputs
                inputs, labels = uu.input_of(data), uu.label_of(data)
                
                # Handle cuda
                inputs, labels = uu.tuple_to(inputs, device), uu.tuple_to(labels, device)
                output = uu.tuple_detach(model(inputs))
                
                # Apply output transforms
                if output_transform is not None:
                    if output_transform.__name__ == 'softmax':
                        output = output_transform(output, dim = 1)
                    else:
                        output = output_transform(output)

                # Apply loss function
                output = loss(output, labels)                

                # Save output for later evaluation
                outbuf.append(output.data.view(-1, 1))
                
        model.train()
        full_output = torch.cat(outbuf, dim=0).to('cpu')
        if type(dataset) == np.ndarray:
            return full_output.numpy()
        else:
            return full_output

    
    
     # Get index of class with max probability
    @staticmethod
    def classify(output, loss_function = None, logger = Logger()):
        """
        Return
        ------
        classes (torch.LongTensor) - Shape [B x 1]
        """
        if isinstance(output, np.ndarray):
            output = torch.from_numpy(output)
        
        if loss_function is not None:
            # Evaluate non-linearity in case of a combined loss-function like CrossEntropy
            if isinstance(loss_function, torch.nn.BCEWithLogitsLoss):
                output = torch.sigmoid(Variable(output)).data
            
            if isinstance(loss_function, torch.nn.CrossEntropyLoss):
                output = F.softmax(Variable(output), dim=1).data

        # Case single output neurons (e.g. one-class-svm sign(output))
        if (output.dim() > 1 and output.size(1) == 1) or output.dim() == 1:
            # makes zeroes positive
            classes = (output - 1e-12).sign().long()  
                
        # One-Hot-Encoding
        if (output.dim() > 1 and output.size(1) > 1):
            classes = output.max(1, keepdim=True)[1]

        return classes
    
    
    @staticmethod
    def compute_accuracy(classes, targets):
        assert targets.shape[0] == classes.shape[0]
              
        # Classification one-hot coded targets are first converted in class labels
        if targets.dim() > 1 and targets.size(1) > 1:
            targets = targets.max(1, keepdim=True)[1]
       
        if not isinstance(targets, torch.LongTensor):
            targets = targets.long()
        
        # number of correctly classified examples
        correct = classes.eq(targets.view_as(classes))
        sum_correct = correct.sum()
        
        if type(sum_correct) == torch.Tensor:
            sum_correct = sum_correct.item()
        
        # accuracy
        accuracy = sum_correct / len(targets)
        return accuracy