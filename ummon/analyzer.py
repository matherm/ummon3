import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .logger import Logger
import ummon.utils as uu
import time
from .predictor import Predictor
from .metrics.accuracy import Accuracy
from .metrics.base import *

__all__ = ["Analyzer", "SupervisedAnalyzer", "ClassificationAnalyzer"]

class Analyzer():
    """
    This class provides a generic analyzer for supervised PyTorch-models. For a given PyTorch-model it 
    computes statistical information about the model, e.g. accuracy, loss, ROC, etc.
    

    Methods
    -------
    evaluate()          : Evaluates a model with given validation dataset
             
    """
    def __init__(self, metrics=None):
        self.name = "ummon.Analyzer"
   
    @staticmethod    
    def evaluate(model, loss_function, dataset, batch_size=-1, compute_accuracy=False, limit_batches=-1, logger=Logger(), metrics=[]):
        """
        Evaluates a model with given validation dataset
        
        Arguments
        ---------
        model           : nn.module
                          The model
        loss_function   : nn.module
                          The loss function to evaluate
        dataset         : torch.utils.data.Dataset OR tuple (X,y, (bs))
                          Dataset to evaluate
        batch_size      : int
                          batch size used for evaluation (default: -1 == ALL)
        compute_accuracy : bool
                          specifies if the output gets classified
        limit_batches   : int
                          specified if only a limited number of batches shall be analyzed. Useful for testing a subset of the training data.
        logger          : ummon.Logger (Optional)
                          The logger to be used for output messages
        metrics:           list (ummon.metrics)
                           A list of metrics that are computed during the evaluation and returned inside the dict
        
        Return
        ------
        Dictionary
        A dictionary containing keys `loss`, `accuracy`, ´samples_per_second`, `detailed_loss`, 'args[]`
        """
        dataloader = uu.gen_dataloader(dataset, batch_size=batch_size, has_labels=True, logger=logger)
        t = time.time()
        loss_average, acc_average = 0., 0.

        if compute_accuracy == True:
            import warnings
            warnings.warn("compute_accuracy==True is deprecated. Use metrics=[Accuracy()] instead.")
            metrics.append(Accuracy())
        
        # initialize the running averages
        compute_online_metrics = len(metrics) > 0
        if compute_online_metrics:
            metrics_dict = {repr(m): 0. for m in metrics if isinstance(m, OnlineMetric)}

        compute_offline_metrics = any([True for m in metrics if isinstance(m, OfflineMetric)])
        if compute_offline_metrics:
            outbuf, labelbuf = [], []        

        use_cuda = next(model.parameters()).is_cuda
        device = "cuda" if use_cuda else "cpu"
        for i, batch in enumerate(dataloader):

            # limit
            if limit_batches == i:
                break

            # data
            inputs, labels = uu.input_of(batch), uu.label_of(batch)
               
            # Handle cuda
            inputs, labels = uu.tuple_to(inputs, device), uu.tuple_to(labels, device)
            output = uu.tuple_detach(model(inputs))

            # Compute output
            loss = loss_function(output, labels)
            loss_average = uu.online_average(loss, i+1, loss_average)

            if compute_online_metrics:
                # Iterates the metrics and updates the running averages
                metrics_dict = {repr(m): uu.online_average(m(output, labels), i+1, metrics_dict[repr(m)]) for m in metrics if isinstance(m, OnlineMetric) }
            
            # Save output for later evaluation
            if compute_offline_metrics:
                outbuf.append(output), labelbuf.append(labels) 

        # NaN check        
        if np.isinf(loss_average):
                raise ValueError("Loss is Inf")

        # NaN check        
        if np.isnan(loss_average):
                raise ValueError("Loss is NaN")


        # save results in dict
        evaluation_dict = {}
        evaluation_dict["accuracy"] = metrics_dict[repr(Accuracy())] if compute_accuracy else 0.
        evaluation_dict["samples_per_second"] = len(dataloader) / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = {"__repr__(loss)" : repr(loss_function)}

        if compute_online_metrics:
            evaluation_dict = {**evaluation_dict, **metrics_dict}
        
        if compute_offline_metrics:
            metrics_dict = {repr(m): m(outbuf, labelbuf) for m in metrics if isinstance(m, OfflineMetric)}
            evaluation_dict = {**evaluation_dict, **metrics_dict}
        
        return evaluation_dict


# Backward compatibility
class SupervisedAnalyzer(Analyzer):
    pass

# Backward compatibility
class ClassificationAnalyzer(SupervisedAnalyzer):
    """
    This class provides a generic analyzer for PyTorch classification models. For a given PyTorch-model it 
    computes statistical information about the model, e.g. accuracy, loss, ROC, etc.
    
    Methods
    -------
    evaluate()          : Evaluates a model with given validation dataset
    """
            
    @staticmethod    
    def evaluate(model, loss_function, dataset, logger=Logger(), batch_size=-1, limit_batches=-1, metrics=[]):
        """
        Evaluates a model with given validation dataset
        
        Arguments
        ---------
        model           : nn.module
                          The model
        loss_function   : nn.module
                          The loss function to evaluate
        dataset         : torch.utils.data.Dataset OR tuple (X,y, (bs))
                          Dataset to evaluate
        logger          : ummon.Logger (Optional)
                          The logger to be used for output messages
        batch_size      : int
                          batch size used for evaluation (default: -1 == ALL)
        limit_batches    : int
                          limits the number of evaluated batches (default: -1 == ALL)
        
        Return
        ------
        Dictionary
        A dictionary containing keys `loss`, `accuracy`, ´samples_per_second`, `detailed_loss`, 'args[]`
        """
        return Analyzer.evaluate(model, loss_function, dataset, logger=logger, limit_batches=limit_batches, batch_size=batch_size, metrics=[Accuracy()] + metrics)