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

__all__ = ["Analyzer", "SupervisedAnalyzer", "ClassificationAnalyzer"]

class Analyzer():
    """
    This class provides a generic analyzer for supervised PyTorch-models. For a given PyTorch-model it 
    computes statistical information about the model, e.g. accuracy, loss, ROC, etc.
    
    
    Methods
    -------
    evaluate()          : Evaluates a model with given validation dataset
    inference()         : Computes model outputs
    compute_roc()       : [Not implemented yet]
             
    """
    def __init__(self):
        self.name = "ummon.Analyzer"
   

    @staticmethod    
    def evaluate(model, loss_function, dataset, batch_size=-1, compute_accuracy=False, limit_batches=-1, logger=Logger()):
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
        
        Return
        ------
        Dictionary
        A dictionary containing keys `loss`, `accuracy`, ´samples_per_second`, `detailed_loss`, 'args[]`
        """
        dataloader = uu.gen_dataloader(dataset, batch_size=batch_size, has_labels=True, logger=logger)
        t = time.time()
        loss_average, acc_average = 0., 0.
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
            loss_average = uu.online_average(loss, i + 1, loss_average)

            if compute_accuracy == True:
                classes = Predictor.classify(output.to('cpu'), loss_function, logger)
                acc = Predictor.compute_accuracy(classes, labels.to('cpu'))
                acc_average = uu.online_average(acc, i + 1, acc_average)

        # NaN check        
        if np.isnan(loss_average):
            raise ValueError("Loss is NaN. Was {}".format(loss_average))

        # save results in dict
        evaluation_dict = {}
        evaluation_dict["accuracy"] = acc_average
        evaluation_dict["samples_per_second"] = len(dataloader) / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = {"__repr__(loss)" : repr(loss_function)}
        
        return evaluation_dict
    
    # output evaluation string for regression
    @staticmethod
    def evalstr(learningstate):
        
        # without validation data
        if learningstate.has_validation_data():
            return 'loss (trn): {:4.5f}, lr={:1.5f}'.format(
                learningstate.current_training_loss(), 
                learningstate.current_lrate())
        
        # with validation data
        else:
            return 'loss(trn/val):{:4.5f}/{:4.5f}, lr={:1.5f}{}'.format(
                learningstate.current_training_loss(), 
                learningstate.current_validation_loss(),
                learningstate.current_lrate(),
                ' [BEST]' if learningstate.is_best_validation_model() else '')


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
    def evaluate(model, loss_function, dataset, logger=Logger(), batch_size=-1, limit_batches=-1):
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
        return Analyzer.evaluate(model, loss_function, dataset, logger=logger, limit_batches=limit_batches, batch_size=batch_size, compute_accuracy=True)

    @staticmethod
    def evalstr(learningstate):
        # without validation data
        if learningstate.has_validation_data():
            return 'loss (trn): {:4.5f}, lr={:1.5f}'.format(
                learningstate.current_training_loss(), 
                learningstate.current_lrate())
        
        # with validation data
        else:
            return 'loss(trn/val):{:4.5f}/{:4.5f}, acc(trn/val):{:.2f}%/{:.2f}%, lr={:1.5f}{}'.format(
                learningstate.current_training_loss(), 
                learningstate.current_validation_loss(),
                learningstate.current_training_acc()*100,
                learningstate.current_validation_acc()*100,
                learningstate.current_lrate(),
                ' [BEST]' if learningstate.is_best_validation_model() else '')