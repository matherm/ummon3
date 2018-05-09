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
from torch.autograd import Variable
from torch.utils.data import DataLoader
import ummon.utils as uu
from .trainer import MetaTrainer
from .analyzer import MetaAnalyzer
from .logger import Logger
from .schedulers import *

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
    def __init__(self, logger, model, loss_function, optimizer, 
                 trainingstate = None,
                 scheduler = None, 
                 model_filename = "model.pth.tar", 
                 model_keep_epochs = False,
                 precision = np.float32,
                 convergence_eps = np.finfo(np.float32).min,
                 combined_training_epochs = 0,
                 use_cuda = False,
                 profile = False):
        super(SupervisedTrainer, self).__init__(logger, model, loss_function, optimizer, trainingstate, 
                 scheduler, 
                 model_filename, 
                 model_keep_epochs,
                 precision,
                 convergence_eps,
                 combined_training_epochs,
                 use_cuda,
                 profile)
        self.analyzer = SupervisedAnalyzer
    
    
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
        # simple interface: training and test data given as numpy arrays
        if type(dataloader_training) == tuple:
            dataset = uu.construct_dataset_from_tuple(logger=self.logger, data_tuple=dataloader_training, train=True)
            # supervised
            if len(dataloader_training) == 3:
                batch = int(dataloader_training[2])
            else:
                self.logger.error('Training data must be provided as a tuple (X,(y),batch) or as PyTorch DataLoader.',
                TypeError)
            dataloader_training = DataLoader(dataset, batch_size=batch, shuffle=True, 
                sampler=None, batch_sampler=None, num_workers=0)
        assert isinstance(dataloader_training, torch.utils.data.DataLoader)
        assert uu.check_precision(dataloader_training.dataset, self.model, self.precision)
        if validation_set is not None:
            if type(validation_set) == tuple or type(validation_set) == np.ndarray:
                validation_set = uu.construct_dataset_from_tuple(logger=self.logger, data_tuple=validation_set, train=False)
            assert isinstance(validation_set, torch.utils.data.Dataset)
            assert uu.check_precision(validation_set, self.model, self.precision)
            
        # COMPUTE BATCHES PER EPOCH
        batches = int(np.ceil(len(dataloader_training.dataset) / dataloader_training.batch_size))
       
        return dataloader_training, validation_set, batches


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
                 dataset = uu.construct_dataset_from_tuple(logger, dataset, train=False)
        
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(model, nn.Module)
        assert uu.check_precision(dataset, model)
        
        use_cuda = next(model.parameters()).is_cuda
        evaluation_dict = {}
        loss_average = 0.
        bs = len(dataset) if batch_size == -1 else batch_size
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, sampler=None, 
                batch_sampler=None)
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
                    after_eval_hook(model, output.data, targets.data, loss.data)
                
                
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
    def __init__(self, logger, model, loss_function, optimizer,
                 trainingstate = None,
                 scheduler = None, 
                 model_filename = "model.pth.tar", 
                 model_keep_epochs = False,
                 precision = np.float32,
                 convergence_eps = np.finfo(np.float32).min,
                 combined_training_epochs = 0,
                 use_cuda = False,
                 profile = False):
        super(ClassificationTrainer, self).__init__(logger, model, loss_function, optimizer, trainingstate,
                 scheduler, 
                 model_filename, 
                 model_keep_epochs,
                 precision,
                 convergence_eps,
                 combined_training_epochs,
                 use_cuda,
                 profile)
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
                 dataset = uu.construct_dataset_from_tuple(logger, dataset, train=False)
        
        assert isinstance(dataset, torch.utils.data.Dataset)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(model, nn.Module)
        assert uu.check_precision(dataset, model)
        
        use_cuda = next(model.parameters()).is_cuda
        evaluation_dict = {}
        
        # Compute Running average training accuracy
        avg_training_acc = 0.
        for saved_output, saved_targets, batch in output_buffer:
            classes = ClassificationAnalyzer.classify(saved_output.cpu())
            acc = ClassificationAnalyzer.compute_accuracy(classes, saved_targets.cpu())
            avg_training_acc = ClassificationAnalyzer._online_average(acc, batch + 1, 
                avg_training_acc)
        
        # evaluate on validation set
        loss_average, acc_average = 0.,0.
        bs = len(dataset) if batch_size == -1 else batch_size
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, sampler=None, batch_sampler=None)
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
                    after_eval_hook(model, output.data, targets.data, loss.data)
                
                # Save output for later evaluation
                outbuf.append((output.data.clone(), targets.data.clone(), i))
                
        # Compute classification accuracy on validation set
        for saved_output, saved_targets, batch in outbuf:
            classes = ClassificationAnalyzer.classify(saved_output.cpu())
            acc = ClassificationAnalyzer.compute_accuracy(classes, saved_targets.cpu())
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
    
    
    # Get index of class with max probability
    @staticmethod
    def classify(output):
        """
        Return
        ------
        classes (torch.LongTensor) - Shape [B x 1]
        """
        
        # Case single output neurons (e.g. one-class-svm sign(output))
        if (output.dim() > 1 and output.size(1) == 1) or output.dim() == 1:
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
    def __init__(self, logger, model, loss_function, optimizer, trainingstate,
                 scheduler = None, 
                 model_filename = "model.pth.tar", 
                 model_keep_epochs = False,
                 precision = np.float32,
                 convergence_eps = np.finfo(np.float32).min,
                 combined_training_epochs = 0,
                 use_cuda = False,
                 profile = False):
        super(SiameseTrainer, self).__init__(logger, model, loss_function, optimizer, trainingstate,
                 scheduler, 
                 model_filename, 
                 model_keep_epochs,
                 precision,
                 convergence_eps,
                 combined_training_epochs,
                 use_cuda,
                 profile)
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
    
    
    def fit(self, dataloader_training, epochs=1, validation_set=None,  
        after_backward_hook=None, after_eval_hook=None, eval_batch_size=-1):
        """
        Fits a model with given training and validation dataset
        
        Arguments
        ---------
        dataloader_training :   torch.utils.data.DataLoader
                                The dataloader that provides the training data
        epochs              :   int
                                Epochs to train
        validation_set      :   torch.utils.data.Dataset
                                The validation dataset
        after_backward_hook :   OPTIONAL function(model, output.data, targets.data, loss.data)
                                A hook that gets called after backward pass during training
        after_eval_hook     :   OPTIONAL function(model, output.data, targets.data, loss.data)
                                A hook that gets called after forward pass during evaluation
        eval_batch_size     :   OPTIONAL int
                                batch size used for evaluation (default: -1 == ALL)
        """
        
        # INPUT VALIDATION
        dataloader_training, validation_set, batches = self._data_validation(dataloader_training, validation_set)
        epochs = self._input_params_validation(epochs)
        
        # PROBLEM SUMMARY
        super(SiameseTrainer, self)._problem_summary(epochs, dataloader_training, validation_set, self.scheduler)
        
        for epoch in range(self.epoch, self.epoch + epochs):
        
            # EPOCH PREPARATION
            time_dict = super(SiameseTrainer, self)._prepare_epoch()
            
            # Moving average
            n, avg_training_loss = 5, None
            training_loss_buffer= np.zeros(n, dtype=np.float32)
            
            # COMPUTE ONE EPOCH                
            for batch, data in enumerate(dataloader_training, 0):
                
                # time dataloader
                time_dict["loader"] = time_dict["loader"] + (time.time() - time_dict["t"])
                
                # Get the inputs
                inputs, targets = data[0], data[1]
                
                # Unfold siamese data
                input_l, input_r, targets = Variable(inputs[0]), Variable(inputs[1]), Variable(targets)
        
                # Handle cuda
                if self.use_cuda:
                    input_l, input_r, targets = input_l.cuda(), input_r.cuda(), targets.cuda()
                
                output, time_dict = super(SiameseTrainer, self)._forward_one_batch((input_l, input_r), time_dict)
                loss,   time_dict = super(SiameseTrainer, self)._loss_one_batch(output, targets, time_dict)
                
                # Backpropagation
                time_dict = super(SiameseTrainer, self)._backward_one_batch(loss, time_dict,
                                                      after_backward_hook, output, targets)
                
                # Loss averaging
                avg_training_loss = self._moving_average(batch, avg_training_loss, loss.cpu(), training_loss_buffer)
                
                # Reporting
                time_dict = super(SiameseTrainer, self)._finish_one_batch(batch, batches, 
                                                    epoch, 
                                                    avg_training_loss,
                                                    dataloader_training.batch_size, 
                                                    time_dict)
        
            # Evaluate
            self._evaluate_training(SiameseAnalyzer, batch, batches, 
                                                          time_dict, 
                                                          epoch, 
                                                          validation_set, 
                                                          avg_training_loss,
                                                          dataloader_training, 
                                                          after_eval_hook, 
                                                          eval_batch_size, None)
    
            # SAVE MODEL
            self.trainingstate.save_state(self.model_filename, self.model_keep_epochs)
            
            # CHECK TRAINING CONVERGENCE
            if super(SiameseTrainer, self)._has_converged():
                break
          
            # ANNEAL LEARNING RATE
            if self.scheduler: 
                try:
                    self.scheduler.step()
                except StepsFinished:
                    break
                
        # DO COMBINED RETRAINING WITH BEST VALIDATION MODEL
        self._combined_retraining(dataloader_training, validation_set, 
            after_backward_hook, after_eval_hook, eval_batch_size)    
       
    
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
        after_eval_hook : OPTIONAL function(model, output.data, targets.data, loss.data)
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
                    after_eval_hook(model, output.data, targets.data, loss.data)
                
                
        evaluation_dict["training_accuracy"] = 0.0        
        evaluation_dict["accuracy"] = 0.0
        evaluation_dict["samples_per_second"] = dataloader.batch_size / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = repr(loss_function)

        return evaluation_dict

       
if __name__ == "__main__":
    pass           
        
