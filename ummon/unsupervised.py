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
from ummon import *
from .trainer import MetaTrainer
from .analyzer import MetaAnalyzer

class UnsupervisedTrainer(MetaTrainer):
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
    scheduler         : torch.optim.lr_scheduler._LRScheduler
                        OPTIONAL A learning rate scheduler
    model_filename    : String
                      : OPTIONAL Name of the persisted model files
    model_keep_epochs : bool
                        OPTIONAL Specifies intermediate (for every epoch) model persistency (default False).
    precision         : np.dtype
                        OPTIONAL Specifiec FP32 or FP64 Training (default np.float32).
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
                 scheduler = None, 
                 model_filename = "model.pth.tar", 
                 model_keep_epochs = False,
                 precision = np.float32,
                 use_cuda = False,
                 profile = False):
           super(UnsupervisedTrainer, self).__init__(logger, model, loss_function, optimizer, 
                 scheduler, 
                 model_filename, 
                 model_keep_epochs,
                 precision,
                 use_cuda,
                 profile)
    
     
    def _input_data_validation_unsupervised(self, dataloader_training, validation_set):
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
            if len(dataloader_training) == 2:
                batch = int(dataloader_training[1])
            else:
                self.logger.error('Training data must be provided as a tuple (X, batch) or as PyTorch DataLoader.',
                TypeError)
            dataloader_training = DataLoader(dataset, batch_size=batch, shuffle=True, 
                sampler=None, batch_sampler=None)
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
    
    
    def fit(self, dataloader_training, epochs=1, validation_set=None, eval_interval=500, 
        trainingstate=None, after_backward_hook=None, after_eval_hook=None, 
        eval_batch_size=-1):
        """
        Fits a model with given training and validation dataset
        
        Arguments
        ---------
        dataloader_training :   torch.utils.data.DataLoader OR tuple (X, batch)
                                The dataloader that provides the training data
        epochs              :   int
                                Epochs to train
        validation_set      :   torch.utils.data.Dataset OR tuple (X)
                                The validation dataset
        eval_interval       :   int
                                Evaluation interval for validation dataset in epochs
        trainingstate       :   ummon.Trainingstate
                                OPTIONAL An optional trainingstate to initialize model and optimizer with an previous state
        after_backward_hook :   OPTIONAL function(model, output.data, targets.data, loss.data)
                                A hook that gets called after backward pass during training
        after_eval_hook     :   OPTIONAL function(model, output.data, targets.data, loss.data)
                                A hook that gets called after forward pass during evaluation
        eval_batch_size     :   OPTIONAL int
                                batch size used for evaluation (default: -1 == ALL)
        
        Return
        ------
        ummon.Trainingstate
        A dictionary containing the trainingstate
        """
        # RESTORE TRAINING STATE
        trainingstate = super(UnsupervisedTrainer, self)._restore_training_state(trainingstate)
        
        # INPUT VALIDATION
        dataloader_training, validation_set, batches = self._input_data_validation_unsupervised(dataloader_training, validation_set)
        epochs, eval_interval = super(UnsupervisedTrainer, self)._input_params_validation(epochs, eval_interval)
        
        # PROBLEM SUMMARY
        super(UnsupervisedTrainer, self)._problem_summary(epochs, dataloader_training, validation_set)
        
        for epoch in range(self.epoch, self.epoch + epochs):
        
            # EPOCH PREPARATION
            time_dict = super(UnsupervisedTrainer, self)._prepare_epoch()
            
            # Moving average
            n, avg_training_loss = 5, None
            training_loss_buffer= np.zeros(n, dtype=np.float32)
            
            # COMPUTE ONE EPOCH                
            for batch, data in enumerate(dataloader_training, 0):
                
                # time dataloader
                time_dict["loader"] = time_dict["loader"] + (time.time() - time_dict["t"])
                
                # Get the inputs
                inputs = Variable(data)
        
                # Handle cuda
                if self.use_cuda:
                    inputs = inputs.cuda()
                
                output, time_dict = super(UnsupervisedTrainer, self)._forward_one_batch(inputs, time_dict)
                loss,   time_dict = super(UnsupervisedTrainer, self)._loss_one_batch(output, inputs, time_dict)
                
                # Backpropagation
                time_dict = super(UnsupervisedTrainer, self)._backward_one_batch(loss, time_dict,
                                                      after_backward_hook, output, inputs)
                
                # Loss averaging
                avg_training_loss = self._moving_average(batch, avg_training_loss, loss.cpu().data[0], training_loss_buffer)
                
                # Reporting
                time_dict = super(UnsupervisedTrainer, self)._finish_one_batch(batch, batches, 
                                                    epoch, 
                                                    avg_training_loss,
                                                    dataloader_training.batch_size, 
                                                    time_dict)
        
            # Evaluate
            super(UnsupervisedTrainer, self)._evaluate_training(UnsupervisedAnalyzer, batch, batches, 
                                                              time_dict, 
                                                              epoch, eval_interval, 
                                                              validation_set, 
                                                              avg_training_loss,
                                                              dataloader_training, 
                                                              after_eval_hook, 
                                                              eval_batch_size, 
                                                              trainingstate)
    
            # SAVE MODEL
            trainingstate.save_state(self.model_filename, self.model_keep_epochs)
                     
        return trainingstate
    
    
    
class UnsupervisedAnalyzer(MetaAnalyzer):
    """
    This class provides a generic analyzer for PyTorch-models. For a given PyTorch-model it 
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
    def evaluate(model, loss_function, dataset, logger=Logger(), after_eval_hook=None, batch_size=-1):
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
        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False, sampler=None, batch_sampler=None)
        for i, data in enumerate(dataloader, 0):
                
                # Take time
                t = time.time()

                # Get the inputs
                inputs = Variable(data)
                
                # Handle cuda
                if use_cuda:
                    inputs = inputs.cuda()
                
                # Execute Model
                output = model(inputs)
                
                # Compute Loss
                loss = loss_function(output, inputs).cpu()
               
                loss_average = Analyzer._online_average(loss.data[0], i + 1, loss_average)
                
                # Run hook
                if after_eval_hook is not None:
                    after_eval_hook(model, output.data, None, loss.data)
                
                
        evaluation_dict["samples_per_second"] = dataloader.batch_size / (time.time() - t)
        evaluation_dict["loss"] = loss_average
        evaluation_dict["detailed_loss"] = repr(loss_function)

        return evaluation_dict
        
if __name__ == "__main__":
    pass           
        