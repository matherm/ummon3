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
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import ummon.utils as uu
from .trainer import MetaTrainer
from .analyzer import MetaAnalyzer
from .logger import Logger
from .schedulers import *
from .predictor import *

__all__ = ["SupervisedTrainer" , "SupervisedAnalyzer", "ClassificationTrainer", "ClassificationAnalyzer", 
           "SiameseTrainer", "SiameseAnalyzer" ]

class SupervisedTrainer(MetaTrainer):
    """
    This class provides a specialized trainer for training supervised PyTorch-models.
    
    Constructor
    -----------
    logger            : ummon.Logger
                        The logger to use (if NULL default logger will be used)
    model             : torch.nn.module
                        The model used for training
    loss_function     : torch.nn.module
                      : The loss function for the optimization
    optimizer         : torch.optim.optimizer
                        The optimizer for the training
    trainingstate     : ummon.Trainingstate
                        A previously instantiated training state variable. Can be used 
                        to initialize model and optimizer with a previous saved state.
    scheduler         : torch.optim.lr_scheduler._LRScheduler
                        OPTIONAL A learning rate scheduler
    model_filename    : String
                      : OPTIONAL Name of the persisted model files
    model_keep_epochs : bool
                        OPTIONAL Specifies intermediate (for every epoch) model persistency (default False).
    precision         : np.dtype
                        OPTIONAL Specifiec FP32 or FP64 Training (default np.float32).
    convergence_eps   : float
                        OPTIONAL Specifies when the training has converged (default np.float32.min).
    combined_training_epochs : int
                        OPTIONAL Specifies how many epochs combined retraining (training and validation data) shall take place 
                            after the usal training cycle (default 0).                        
    use_cuda          : bool
                        OPTIONAL Shall cuda be used as computational backend (default False)
    profile           : bool
                        OPTIONAL Activates some advanced timing and profiling logs (default False)
    
    Methods
    -------
    fit()            :  trains a model
    _evaluate()      :  validates the model
    _moving_average():  helper method
             
    """
    def __init__(self, *args, **kwargs):
        super(SupervisedTrainer, self).__init__(*args, **kwargs)
        self.analyzer = SupervisedAnalyzer
    
    
    def _data_validation(self, dataloader_training, validation_set):
        """
        Does input data validation for training and validation data.
        
        Arguments
        ---------
        *dataloader_training (torch.utils.data.Dataloader OR
                              torch.utils.data.Dataset OR 
                              numpy (X, y, bs) OR 
                              torch.Tensor (X y,, bs) : A data structure holding the training data.
        *validation_set (torch.utils.data.Dataset) : A dataset holding the validation data
        
        Return
        ------
        *dataloader_training (torch.utils.data.Dataloader) : Same as input or corrected versions from input.
        *validation_set (torch.utils.data.Dataset) : Same as input or corrected versions from input.
        *batches (int) : Computed total number of training batches.
        """
        # simple interface: training and test data given as numpy arrays
        if type(dataloader_training) == tuple:
            data = dataloader_training
            if len(data) == 3:
                batch_size = int(data[2])
            else:
                self.logger.error('Training data must be provided as a tuple (X, y, batch) or as PyTorch DataLoader.',TypeError)
            if isinstance(data[0], np.ndarray) or uu.istensor(data[0]):
                torch_dataset = uu.construct_dataset_from_tuple(self.logger, data, train=True)

        if isinstance(dataloader_training, torch.utils.data.Dataset):
                batch_size = -1
                torch_dataset = dataloader_training

        if isinstance(dataloader_training, torch.utils.data.DataLoader):
            dataloader = dataloader_training
            torch_dataset = dataloader.dataset
        else:
            bs = len(torch_dataset) if batch_size == -1 else batch_size
            dataloader = DataLoader(torch_dataset, batch_size=bs, shuffle=False, sampler=None, batch_sampler=None)       

        assert uu.check_precision(dataloader.dataset, self.model, self.precision)
        
        # COMPUTE BATCHES PER EPOCH
        batches = int(np.ceil(len(torch_dataset) / dataloader.batch_size))
       
        if validation_set is not None:
            if type(validation_set) == tuple or type(validation_set) == np.ndarray:
                validation_set = uu.construct_dataset_from_tuple(logger=self.logger, data_tuple=validation_set, train=False)
            assert isinstance(validation_set, torch.utils.data.Dataset)
            assert uu.check_precision(validation_set, self.model, self.precision)
            
        return dataloader, validation_set, batches


class SupervisedAnalyzer(MetaAnalyzer):
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
        self.name = "ummon.SupervisedAnalyzer"
            
    @staticmethod
    def _data_validation(eval_dataset, batch_size, logger):
        """
        Does input data validation for training and validation data.
        
        Arguments
        ---------
        *dataset (torch.utils.data.Dataloader OR
                              torch.utils.data.Dataset OR 
                              numpy (X, y) OR 
                              torch.Tensor (X, y) : A data structure holding the validation data.
        *batch_size (int) : The batch size
        *logger (ummon.logger) : the logger 
        
        Return
        ------
        *dataloader (torch.utils.data.Dataloader) : Same as input or corrected versions from input.
        """
        # simple interface: training and test data given as numpy arrays
        if type(eval_dataset) == tuple:
            data = eval_dataset
            if len(eval_dataset) != 2:
                logger.error('Training data must be provided as a tuple (X, y) or as PyTorch DataLoader.',TypeError)
            if isinstance(data[0], np.ndarray) or uu.istensor(data[0]):
                torch_dataset = uu.construct_dataset_from_tuple(logger, data, train=False)

        if isinstance(eval_dataset, torch.utils.data.Dataset):
                torch_dataset = eval_dataset

        if isinstance(eval_dataset, torch.utils.data.DataLoader):
            dataloader = eval_dataset
            torch_dataset = dataloader.dataset
        else:
            bs = len(torch_dataset) if batch_size == -1 else batch_size
            dataloader = DataLoader(torch_dataset, batch_size=bs, shuffle=False, sampler=None, batch_sampler=None)        

        return dataloader
    
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
        after_eval_hook : OPTIONAL function(output.data, targets.data, loss.data)
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
        loss_average = 0.
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
                output = model(Variable(inputs))
                
                # Compute Loss
                targets = Variable(targets)
                loss = loss_function(output, targets).cpu()
               
                loss_average = MetaAnalyzer._online_average(loss, i + 1, loss_average)
                
                # Run hook
                if after_eval_hook is not None:
                    after_eval_hook(output.data, targets.data, loss.data)
                
                
        evaluation_dict["training_accuracy"] = 0.0        
        evaluation_dict["accuracy"] = 0.0
        evaluation_dict["samples_per_second"] = dataloader.batch_size / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = repr(loss_function)
        
        return evaluation_dict
    
    
    # output evaluation string for regression
    @staticmethod
    def evalstr(learningstate):
        
        # without validation data
        if learningstate.state["validation_loss[]"] == []:
            return 'loss (trn): {:4.5f}, lr={:1.5f}'.format(
                learningstate.state["training_loss[]"][-1][1], 
                learningstate.state["lrate[]"][-1][1])
        
        # with validation data
        else:
            is_best = learningstate.state["validation_loss[]"][-1][1] == \
                learningstate.state["best_validation_loss"][1]
            return 'loss(trn/val):{:4.5f}/{:4.5f}, lr={:1.5f}'.format(
                learningstate.state["training_loss[]"][-1][1], 
                learningstate.state["validation_loss[]"][-1][1],
                learningstate.state["lrate[]"][-1][1],
                ' [BEST]' if is_best else '')


class ClassificationTrainer(SupervisedTrainer):
    """
    This class provides a specialized trainer for training Regression PyTorch-models.
    
    Constructor
    -----------
    logger            : ummon.Logger
                        The logger to use (if NULL default logger will be used)
    model             : torch.nn.module
                        The model used for training
    loss_function     : torch.nn.module
                      : The loss function for the optimization
    optimizer         : torch.optim.optimizer
                        The optimizer for the training
    trainingstate     : ummon.Trainingstate
                        A previously instantiated training state variable. Can be used 
                        to initialize model and optimizer with a previous saved state.
    scheduler         : torch.optim.lr_scheduler._LRScheduler
                        OPTIONAL A learning rate scheduler
    model_filename    : String
                      : OPTIONAL Name of the persisted model files
    model_keep_epochs : bool
                        OPTIONAL Specifies intermediate (for every epoch) model persistency (default False).
    precision         : np.dtype
                        OPTIONAL Specifiec FP32 or FP64 Training (default np.float32).
    convergence_eps   : float
                        OPTIONAL Specifies when the training has converged (default np.float32.min).
    combined_training_epochs : int
                        OPTIONAL Specifies how many epochs combined retraining (training and validation data) shall take place 
                            after the usal training cycle (default 0).                        
    use_cuda          : bool
                        OPTIONAL Shall cuda be used as computational backend (default False)
    profile           : bool
                        OPTIONAL Activates some advanced timing and profiling logs (default False)
    
    Methods
    -------
    fit()            :  trains a model
    _evaluate()      :  validates the model
    _moving_average():  helper method
             
    """
    def __init__(self, *args, **kwargs):
        super(ClassificationTrainer, self).__init__(*args, **kwargs)
        self.analyzer = ClassificationAnalyzer


class ClassificationAnalyzer(SupervisedAnalyzer):
    """
    This class provides a generic analyzer for PyTorch classification models. For a given PyTorch-model it 
    computes statistical information about the model, e.g. accuracy, loss, ROC, etc.
    
    
    Methods
    -------
    evaluate()          : Evaluates a model with given validation dataset
    classify()          : Classifies model outputs with One-Hot-Encoding            
    compute_accuracy()  : Computes the accuracy of a classification
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
        after_eval_hook : OPTIONAL function(output.data, targets.data, loss.data)
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
        
        # Compute Running average training accuracy
        if output_buffer is not None and len(output_buffer) > 0:
            avg_training_acc = 0.
            for saved_output, saved_targets, batch in output_buffer:
                classes = Predictor.classify(saved_output.cpu(), loss_function, logger)
                acc = Predictor.compute_accuracy(classes, saved_targets.cpu())
                avg_training_acc = MetaAnalyzer._online_average(acc, batch + 1, 
                    avg_training_acc)
        else:
            avg_training_acc = None
        
        # evaluate on validation set
        use_cuda = next(model.parameters()).is_cuda
        evaluation_dict = {}
        loss_average, acc_average = 0.,0.
        outbuf = []
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
                output = model(Variable(inputs))
                
                # Compute Loss
                targets = Variable(targets)

                loss = loss_function(output, targets).cpu()
               
                loss_average = MetaAnalyzer._online_average(loss, i + 1, loss_average)
                
                # Run hook
                if after_eval_hook is not None:
                    after_eval_hook(output.data, targets.data, loss.data)
                
                # Save output for later evaluation
                outbuf.append((output.data.clone(), targets.data.clone(), i))
                
        # Compute classification accuracy on validation set
        for saved_output, saved_targets, batch in outbuf:
            classes = Predictor.classify(saved_output.cpu(), loss_function, logger)
            acc = Predictor.compute_accuracy(classes, saved_targets.cpu())
            acc_average = ClassificationAnalyzer._online_average(acc, batch + 1, acc_average)
        
        # save results in dict
        evaluation_dict["training_accuracy"] = avg_training_acc
        evaluation_dict["accuracy"] = acc_average
        evaluation_dict["samples_per_second"] = dataloader.batch_size / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = repr(loss_function)
        evaluation_dict["args[]"] = {}
        
        del outbuf[:]
        return evaluation_dict
    
    
    # output evaluation string for classifier
    @staticmethod
    def evalstr(learningstate):
        
        # without validation data
        if learningstate.state["validation_loss[]"] == []:
            return 'loss (trn): {:4.5f}, lr={:1.5f}'.format(
                learningstate.state["training_loss[]"][-1][1], 
                learningstate.state["lrate[]"][-1][1])
        
        # with validation data
        else:
            is_best = learningstate.state["validation_loss[]"][-1][1] == \
                learningstate.state["best_validation_loss"][1]
            return 'loss(trn/val):{:4.5f}/{:4.5f}, acc(val):{:.2f}%, lr={:1.5f}{}'.format(
                learningstate.state["training_loss[]"][-1][1], 
                learningstate.state["validation_loss[]"][-1][1],
                learningstate.state["validation_accuracy[]"][-1][1]*100,
                learningstate.state["lrate[]"][-1][1],
                ' [BEST]' if is_best else '')


class SiameseTrainer(SupervisedTrainer):
    """
    This class provides a specialized trainer for training Regression PyTorch-models.
    
    Constructor
    -----------
    logger            : ummon.Logger
                        The logger to use (if NULL default logger will be used)
    model             : torch.nn.module
                        The model used for training
    loss_function     : torch.nn.module
                      : The loss function for the optimization
    optimizer         : torch.optim.optimizer
                        The optimizer for the training
    trainingstate     : ummon.Trainingstate
                        A previously instantiated training state variable. Can be used 
                        to initialize model and optimizer with a previous saved state.
    scheduler         : torch.optim.lr_scheduler._LRScheduler
                        OPTIONAL A learning rate scheduler
    model_filename    : String
                      : OPTIONAL Name of the persisted model files
    model_keep_epochs : bool
                        OPTIONAL Specifies intermediate (for every epoch) model persistency (default False).
    precision         : np.dtype
                        OPTIONAL Specifiec FP32 or FP64 Training (default np.float32).
    convergence_eps   : float
                        OPTIONAL Specifies when the training has converged (default np.float32.min).
    combined_training_epochs : int
                        OPTIONAL Specifies how many epochs combined retraining (training and validation data) shall take place 
                            after the usal training cycle (default 0).                        
    use_cuda          : bool
                        OPTIONAL Shall cuda be used as computational backend (default False)
    profile           : bool
                        OPTIONAL Activates some advanced timing and profiling logs (default False)
    
    Methods
    -------
    fit()            :  trains a model
    _evaluate()      :  validates the model
    _moving_average():  helper method
             
    """
    def __init__(self,  *args, **kwargs):
        super(SiameseTrainer, self).__init__(*args, **kwargs)
        self.analyzer = SiameseAnalyzer
    
    
    def _data_validation(self, dataloader_training, validation_set):
        """
        Does input data validation for training and validation data.
        
        Arguments
        ---------
        *dataloader_training (torch.utils.data.Dataloader) : A dataloader holding the training data.
        *validation_set (torch.utils.data.Dataset) : A dataset holding the validation data
        
        Return
        ------
        *dataloader_training (torch.utils.data.Dataloader) : Same as input or corrected versions from input.
        *validation_set (torch.utils.data.Dataset) : Same as input or corrected versions from input.
        *batches (int) : Computed total number of training batches.
        """
        assert isinstance(dataloader_training, torch.utils.data.DataLoader)
        assert uu.check_precision(dataloader_training.dataset, self.model, self.precision)
        if validation_set is not None:
            assert isinstance(validation_set, torch.utils.data.Dataset)
            assert uu.check_precision(validation_set, self.model, self.precision)
            
        # COMPUTE BATCHES PER EPOCH
        batches = int(np.ceil(len(dataloader_training.dataset) / dataloader_training.batch_size))
       
        return dataloader_training, validation_set, batches
    
    
    # prepares one batch for processing
    def _get_batch(self, data):
        
        # Get the inputs
        inputs, targets = data[0], data[1]
        
        # Unfold siamese data
        input_l, input_r, targets = Variable(inputs[0]), Variable(inputs[1]), Variable(targets)
        
        # Handle cuda
        if self.use_cuda:
            input_l, input_r, targets = input_l.cuda(), input_r.cuda(), targets.cuda()
        
        return (input_l, input_r), targets


class SiameseAnalyzer(SupervisedAnalyzer):
    """
    This class provides a generic analyzer for PyTorch siamese models. For a given model it 
    computes statistical information about the model, e.g. accuracy, loss, ROC, etc.
    
    
    Methods
    -------
    evaluate()          : Evaluates a model with given validation dataset
    inference()         : Computes model outputs
    compute_roc()       : [Not implemented yet]
             
    """
    def __init__(self):
        self.name = "ummon.SiameseAnalyzer"
            
            
    @staticmethod    
    def evaluate(model, loss_function, dataset, logger=Logger(), after_eval_hook=None, 
        batch_size=-1, output_buffer=None):
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
        logger          : ummon.Logger (Optional)
                          The logger to be used for output messages
        after_eval_hook : OPTIONAL function(output.data, targets.data, loss.data)
                          A hook that gets called after forward pass
        batch_size      : int
                          batch size used for evaluation (default: -1 == ALL)
        
        Return
        ------
        Dictionary
        A dictionary containing keys `loss`, `accuracy`, ´samples_per_second`, `detailed_loss`, 'args[]`
        """
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(model, nn.Module)
        assert uu.check_precision(dataset, model)
        
        use_cuda = next(model.parameters()).is_cuda
        evaluation_dict = {}
        loss_average = 0.
        bs = len(dataset) if batch_size == -1 else batch_size
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, sampler=None, batch_sampler=None)
        for i, data in enumerate(dataloader, 0):
                
                # Take time
                t = time.time()

                # Get the inputs
                inputs, targets = data[0], data[1]
                
                # Unfold siamese data
                input_l, input_r, targets = Variable(inputs[0]), Variable(inputs[1]), Variable(targets)
        
                 # Handle cuda
                if use_cuda:
                    input_l, input_r, targets = input_l.cuda(), input_r.cuda(), targets.cuda()
                
                # Execute Model
                output = model((input_l, input_r))
                
                # Compute Loss
                loss = loss_function(output, targets).cpu()
               
                loss_average = MetaAnalyzer._online_average(loss, i + 1, loss_average)
                
                # Run hook
                if after_eval_hook is not None:
                    after_eval_hook(output.data, targets.data, loss.data)
                
                
        evaluation_dict["training_accuracy"] = 0.0        
        evaluation_dict["accuracy"] = 0.0
        evaluation_dict["samples_per_second"] = dataloader.batch_size / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = repr(loss_function)

        return evaluation_dict

       
if __name__ == "__main__":
    pass           
        
