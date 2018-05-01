import time
import numpy as np
import torch.nn as nn
import torch.utils.data
import ummon.utils as uu
from torch.autograd import Variable
from torch.utils.data import DataLoader
from .logger import Logger
from .schedulers import StepLR_earlystop
from .trainingstate import *

__all__ = ["MetaTrainer"]

class MetaTrainer:
    """
    This class provides a generic trainer for training PyTorch-models.
    For specialized use in Regression or Classification this class needs to be subclassed.
    
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
                 combined_training_epochs = False,
                 use_cuda = False,
                 profile = False):
        
        assert type(logger) == Logger
        assert isinstance(model, nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(loss_function, nn.Module)
        if trainingstate == None:
            self.trainingstate = Trainingstate()
        elif not isinstance(trainingstate, Trainingstate):
            raise TypeError('{} is not a training state'.format(type(trainingstate).__name__))
        else:
            self.trainingstate = trainingstate
        if scheduler is not None:
            if not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) and not \
                isinstance(scheduler, StepLR_earlystop):
                raise TypeError('{} is not a scheduler'.format(type(scheduler).__name__))
            if isinstance(scheduler, StepLR_earlystop) and trainingstate == None:
                raise ValueError('StepLR_earlystop needs an external Trainingstate (you provided None).')
        assert precision == np.float32 or precision == np.float64
        
        # MEMBER VARIABLES
        
        # Training parameters
        self.logger = logger
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epoch = 0
        self.precision = precision
        self.convergence_eps = convergence_eps
        self.combined_training_epochs = combined_training_epochs
        self.use_cuda = use_cuda
        self.profile = profile
        
        # training state was filled by previous training or by persisted training state
        if self.trainingstate.state is not None:
            self._status_summary()
            self.epoch = self.trainingstate.state["training_loss[]"][-1][0]
            self.optimizer = self.trainingstate.load_optimizer(self.optimizer)
            self.model = self.trainingstate.load_weights(self.model, optimizer)
            if isinstance(self.scheduler, StepLR_earlystop):
                self.scheduler = self.trainingstate.load_scheduler(self.scheduler)
        
        #  Persistency parameters
        self.model_filename = model_filename.split(Trainingstate().extension)[0]
        self.model_keep_epochs = model_keep_epochs
        
        # Computational configuration
        if self.use_cuda:
            if not torch.cuda.is_available():
                logger.error('CUDA is not available on your system.')
        self.model = Trainingstate.transform_model(self.model, self.optimizer, precision, use_cuda)
    
    
    def fit(self):
        raise NotImplementedError("This class is superclass.")
    
    def _prepare_epoch(self):
        """
        Some epoch initialization stuff like adjusting the learning rate scheduler and initializing the clock.
        
        Return
        ------
        time_dict (dictionary of timestamps): Dictionary that is used for profiling executing time.
        
        """
        # TAKE TIME
        time_dict = {"t"        : time.time(),
                     "loader"   : 0,
                     "model"    : 0,
                     "loss"     : 0,
                     "backprop" : 0,
                     "hooks"    : 0,
                     "total"    : 0}
        
        
        return time_dict

        
    def _forward_one_batch(self, inputs, time_dict):
        """
        Some epoch initialization stuff like adjusting the learning rate scheduler and initializing the clock.
        
        Arguments
        ---------
        *inputs    (torch.autograd.Variable): A packed torch.Tensor representing a single mini-batch.
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        Return
        ------
        *output    (torch.autograd.Variable): A packed torch.Tensor representing a single output of a mini-batch.
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        """
        # Execute Model
        output = self.model(inputs)
        
        # time model
        if self.profile and self.use_cuda: torch.cuda.synchronize()
        time_dict["model"] = time_dict["model"] + (time.time() - time_dict["t"])
       
        return output, time_dict
       
    def _loss_one_batch(self, output, targets, time_dict):
        """
        Computes the loss for a single mini-batch
        
        Arguments
        ---------
        *output    (torch.autograd.Variable): A packed torch.Tensor representing a single output of a mini-batch.
        *targets   (torch.autograd.Variable): The targets for a mini-batch
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        Return
        ------
        *loss      (torch.autograd.Variable): The computed loss as scalar
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        """
        assert type(output) == torch.autograd.Variable

        if targets.is_cuda or output.is_cuda:
            output, targets = output.cuda(), targets.cuda()
        
        loss = self.criterion(output, targets)
        
        # time loss
        if self.profile and self.use_cuda: torch.cuda.synchronize()
        time_dict["loss"] = time_dict["loss"] + (time.time() - time_dict["t"])
        return loss, time_dict
      
    def _backward_one_batch(self, loss, time_dict, after_backward_hook=None, output=None, targets=None):
        """
        Computes the loss for a single mini-batch
        
        Arguments
        ---------
        *loss      (torch.autograd.Variable): The computed loss as scalar
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        *OPTIONAL after_backward_hook (function(model, output.data, targets.data, loss.data)):
                                            : A hook that gets executed after backward pass.
        *OPTIONAL output    (torch.autograd.Variable): A packed torch.Tensor representing a 
                                              single output of a mini-batch.
        *OPTIONAL targets   (torch.autograd.Variable): The targets for a mini-batch
        
        Return
        ------
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        """
        # Zero the gradient    
        self.optimizer.zero_grad()
        
        # Backpropagation
        loss.backward()
        
        # time backprop
        if self.profile and self.use_cuda: torch.cuda.synchronize()
        time_dict["backprop"] = time_dict["backprop"] + (time.time() - time_dict["t"])
        
        # Run hooks
        if after_backward_hook is not None:
            after_backward_hook(self.model, output.data, targets.data, loss.cpu().data)
        
        # time hooks
        if self.profile and self.use_cuda: torch.cuda.synchronize()
        time_dict["hooks"] = time_dict["hooks"] + (time.time() - time_dict["t"])
        
        # Take gradient descent
        self.optimizer.step()
        
        return time_dict

    def _finish_one_batch(self, batch, batches, epoch, avg_training_loss, training_batchsize, time_dict):
        """
        Finishes a batch and updates moving average loss. Last it logs the current state.
        
        Arguments
        ---------
        *batch  (int) : The current batch number.
        *batches (int) : The total batch number.
        *epoch (int) : The current epoch number.
        *avg_training_loss (float) : The current training loss.
        *training_batchsize (int) ; The batch size of the scheduled training.
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        Return
        ------
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        """
        # total time
        if self.profile and self.use_cuda: torch.cuda.synchronize()
        time_dict["total"] = time_dict["total"] + (time.time() - time_dict["t"])
        
        # Log status
        self.logger.log_one_batch(epoch + 1, batch + 1, batches, avg_training_loss, training_batchsize, time_dict, self.profile)
        
        # Reset time
        time_dict["t"] = time.time()
        return time_dict
        

    def _moving_average(self, t, ma, value, buffer):
        """
        Helper method for computing moving averages.
        
        Arguments
        ---------
        * t (int) : The timestep
        * ma (float) : Current moving average
        * value (float) : Current value
        * buffer (List<float>) : The buffer of size N
        
        Return
        ------
        * moving_average (float) : The new computed moving average.
        
        """
        n = buffer.shape[0]
        if ma is None:
            moving_average = value
            buffer += value
        else:
            moving_average = ma + (value / n) - (buffer[t % n] / n)
        buffer[t % n] = value
        return moving_average
    
    
    def _input_params_validation(self, epochs, eval_interval):
        """
        Validates the given parameters
        
        Arguments
        ---------
        *epochs (int) : The number of scheduled epochs.
        *eval_interval (int) : The interval between model evaluation with validation dataset
        
        Return
        ------
        *epochs (int) : Same as input or corrected versions of input.
        *eval_interval(int) : Same as input or corrected versions of input.
        """
        # check parameters
        epochs = int(epochs)
        if epochs < 1:
            self.logger.error('Number of epochs must be > 0.', ValueError)
        eval_interval = int(eval_interval)
        
        return epochs, eval_interval     
    
    
    def _has_converged(self):
        """
        Checks if the training has converged. 
        Criterium is specified by self.convergence_eps
        
        Return
        ------
        *break_training (bool) : 
        """
        if len(self.trainingstate["training_loss[]"]) > 2:
            if np.abs(self.trainingstate["training_loss[]"][-1][1] - self.trainingstate["training_loss[]"][-2][1]) < self.convergence_eps:
                self.logger.info("Training has converged. Epsilon was {:.2e}".format(self.convergence_eps))
                return True
        
        return False
    
    
    def _problem_summary(self, epochs, dataloader_training, validation_set, scheduler = None):
        """
        Prints the problem summary
        
        Arguments
        ---------
        *epochs (int) : The number of scheduled epochs.
        *dataloader_training (torch.utils.data.Dataloader) : A dataloader holding the training data.
        *validation_set (torch.utils.data.Dataset) : A dataset holding the validation data
        """
        # PRINT SOME INFORMATION ABOUT THE SCHEDULED TRAINING
        early_stopping = isinstance(scheduler, StepLR_earlystop)
        self.logger.print_problem_summary(self.model, self.criterion, self.optimizer, 
            dataloader_training, validation_set, epochs, early_stopping, self.combined_training_epochs)
        
        # training startup message
        self.logger.info('Begin training: {} epochs.'.format(epochs))    
        
        
    def _status_summary(self):
        if self.trainingstate.state is not None:
            self.logger.info("[Status]" )
            trs = self.trainingstate.get_summary()
            self.logger.info('Epochs: {}, best training loss ({}): {:.4f}, best validation loss ({}): {:.4f}'.format(
                trs['Epochs'], trs['Best Training Loss'][0], trs['Best Training Loss'][1], 
                trs['Best Validation Loss'][0], trs['Best Validation Loss'][1]))
    
    
    def _evaluate_training(self, Analyzer, batch, batches, 
                  time_dict, 
                  epoch, eval_interval, 
                  validation_set, 
                  avg_training_loss,
                  dataloader_training, 
                  after_eval_hook, 
                  eval_batch_size):
        """
        Evaluates the current training state against agiven analyzer
        
        Arguments
        ---------
        *Analyzer (ummon.Analyzer) : A training type specific analyzer.
        *batch (int) : The batch number.
        *batches (int) : The total number of batches.
        *time_dict (dict) : Dictionary that is used for profiling executing time.
        *epoch (int) : The current epoch.
        *eval_interval (int) : The interval in epochs for evaluation against validation dataset.
        *validation_set (torch.utils.data.Dataset) : Validation data.
        *avg_training_loss (float) : the current training loss
        *dataloader_training : The training data
        *after_eval_hook : A hook that gets executed after evaluation
        *eval_batch_size (int) : A custom batch size to be used during evaluation
        
        """
        
        # Log epoch
        self.logger.log_epoch(epoch + 1, batch + 1, 
                              batches, 
                              avg_training_loss, 
                              dataloader_training.batch_size, 
                              time_dict, 
                              self.profile)
        
        if (epoch +1) % eval_interval == 0 and validation_set is not None:
            
                # MODEL EVALUATION
                evaluation_dict = Analyzer.evaluate(self.model, 
                                                    self.criterion, 
                                                    validation_set, 
                                                    self.logger, 
                                                    after_eval_hook, 
                                                    batch_size=eval_batch_size)
                
                # UPDATE TRAININGSTATE
                self.trainingstate.update_state(epoch + 1, self.model, self.criterion, self.optimizer, 
                                        training_loss = avg_training_loss, 
                                        training_batchsize = dataloader_training.batch_size,
                                        training_dataset = dataloader_training.dataset,
                                        trainer_instance = type(self),
                                        precision = self.precision,
                                        detailed_loss = repr(self.criterion),
                                        validation_loss = evaluation_dict["loss"], 
                                        validation_dataset = validation_set,
                                        samples_per_second = evaluation_dict["samples_per_second"],
                                        scheduler = self.scheduler,
                                        combined_retraining = self.combined_training_epochs)
        
                self.logger.log_regression_evaluation(self.trainingstate, self.profile)
                        
        else: # no validation set
            self.trainingstate.update_state(epoch + 1, self.model, self.criterion, self.optimizer, 
                training_loss = avg_training_loss, 
                training_batchsize = dataloader_training.batch_size,
                training_dataset = dataloader_training.dataset,
                trainer_instance = type(self),
                precision = self.precision,
                detailed_loss = repr(self.criterion))
    
    
    def _combined_retraining(self, dataloader_training, validation_set, 
                             eval_interval, after_backward_hook, after_eval_hook, eval_batch_size):
        """
        Does combined retraining with validation AND training data. Can be used after the normal training to refine the model.
        
        Arguments
        ---------
        dataloader_training :   torch.utils.data.DataLoader OR tuple (X, batch)
                                The dataloader that provides the training data
        validation_set      :   torch.utils.data.Dataset OR tuple (X)
                                The validation dataset
        eval_interval       :   int
                                Evaluation interval for validation dataset in epochs
        after_backward_hook :   OPTIONAL function(model, output.data, targets.data, loss.data)
                                A hook that gets called after backward pass during training
        after_eval_hook     :   OPTIONAL function(model, output.data, targets.data, loss.data)
                                A hook that gets called after forward pass during evaluation
        eval_batch_size     :   OPTIONAL int
                                batch size used for evaluation (default: -1 == ALL)
        
        """
        if self.combined_training_epochs > 0:
            if validation_set is None:
                self.logger.warn("Combined retraining needs validation data.")
            else:
                # load best validation model
                self.model = self.trainingstate.load_weights_best_validation(self.model, self.optimizer)
                
                # combine the two datasets
                dataloader_combined = uu.add_dataset_to_loader(dataloader_training, validation_set)   
                
                # give some information about what we are going to do
                self.logger.info('Begin combined retraining: {} epochs.'.format(self.combined_training_epochs))  
                
                # get current state
                combined_training_epochs = self.combined_training_epochs
                model_filename = self.model_filename
                model_keep_epochs = self.model_keep_epochs
                
                # modify state so that recursion is not infinite, and filenames are correct
                self.combined_training_epochs = 0
                self.model_filename = str(self.model_filename + self.trainingstate.combined_retraining_pattern)
                self.model_keep_epochs = True
                
                # do actual retraining
                self.fit(dataloader_combined, 
                         epochs=combined_training_epochs, 
                         validation_set=None, 
                         eval_interval=eval_interval, 
                         after_backward_hook=after_backward_hook, 
                         after_eval_hook=after_eval_hook, 
                         eval_batch_size=eval_batch_size)
                
                # restore previous state
                self.combined_training_epochs = combined_training_epochs
                self.model_filename = model_filename
                self.model_keep_epochs = model_keep_epochs

    def _repair_references(self, model, optimizer, scheduler):
        """
        Helper method to repair references after the model's parameters have changed.
        This happens when the model is converted to CUDA or older weights are loaded.
        When this happens, the optimizer optimizes old weights as he does not have the current weights.
        Therefore we need to repoint the optimizers weights to the new model.
        
        Arguments
        --------
        model (nn.module) : the new model with weights
        optimizer (torch.utils.data.optimizer) : the optimizer
        scheduler (torch.utils.data.scheduler) : OPTIONAL a scheduler pointing to an optimizer

        """
        pass
