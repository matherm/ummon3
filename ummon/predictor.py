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
        # Avoid getting problems when using torch_geometric.data.DataLoader
        try:
            import torch_geometric
            if not isinstance(dataset, torch_geometric.data.DataLoader):
                assert uu.check_precision(DataLoader, model)
        except NameError:
            assert uu.check_precision(dataloader, model)

        use_cuda = next(model.parameters()).is_cuda
        device = "cuda" if use_cuda else "cpu"
        
        is_training = model.training
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
                outbuf.append(output)
        
        if is_training:
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
        
        is_training = model.training
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
        
        if is_training:
            model.train()                
        model.train()
        full_output = torch.cat(outbuf, dim=0).to('cpu')
        if type(dataset) == np.ndarray:
            return full_output.numpy()
        else:
            return full_output
    
     # Get index of class with max probability
    @staticmethod
    def classify(output, loss_function = None, logger = Logger()):
        from ummon.metrics.accuracy import classify
        import warnings
        warnings.warn("Deprecated Predictor.classify(). Please use ummon.metrics.accuracy.classify()")
        return classify(output, loss_function = None, logger = Logger())
    
    
    @staticmethod
    def compute_accuracy(classes, targets):
        from ummon.metrics.accuracy import compute_accuracy
        import warnings
        warnings.warn("Deprecated Predictor.compute_accuracy(). Please use ummon.metrics.accuracy.compute_accuracy()")
        return compute_accuracy(classes, targets)