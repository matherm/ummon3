import time
import numpy as np
import torch.nn as nn
import torch.utils.data
import ummon.utils as uu
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from .logger import Logger
from .schedulers import *
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
    def __init__(self, *args, **kwargs):
        
        # required arguments
        if len(args) != 4:
            raise ValueError('You must provide at least a logger, model, loss_function and optimizer for training.') 
        self.logger = args[0]
        assert type(self.logger) == Logger
        self.model = args[1]
        assert isinstance(self.model, nn.Module)
        self.criterion = args[2]
        assert isinstance(self.criterion, nn.Module)
        self.optimizer = args[3]
        assert isinstance(self.optimizer, torch.optim.Optimizer)
        
        # defaults
        self.trainingstate = Trainingstate()
        self.scheduler = None
        self.analyzer = None # needs to be implemented by subclass
        self.model_filename = "model.pth.tar"
        self.model_keep_epochs = False
        self.precision = np.float32
        self.convergence_eps = np.finfo(np.float32).min
        self.combined_training_epochs = False
        self.use_cuda = False
        self.profile = False
        
        # optional arguments
        for key in kwargs:
            if key == 'trainingstate':
                if not isinstance(kwargs[key], Trainingstate):
                    raise TypeError('{} is not a training state'.format(type(kwargs[key]).__name__))
                self.trainingstate = kwargs[key]
            elif key == 'scheduler':
                if not isinstance(kwargs[key], torch.optim.lr_scheduler._LRScheduler) and not \
                    isinstance(kwargs[key], StepLR_earlystop):
                    raise TypeError('{} is not a scheduler'.format(type(kwargs[key]).__name__))
                if isinstance(kwargs[key], StepLR_earlystop) and 'trainingstate' not in kwargs.keys():
                    raise ValueError('StepLR_earlystop needs an external Trainingstate (you provided None).')
                self.scheduler = kwargs[key]
            elif key == 'model_filename':
                model_filename = str(kwargs[key])
                self.model_filename = model_filename.split(self.trainingstate.extension)[0]
            elif key == 'model_keep_epochs':
                self.model_keep_epochs = int(kwargs[key])
            elif key == 'precision':
                assert kwargs[key] == np.float32 or kwargs[key] == np.float64
                self.precision = kwargs[key]
            elif key == 'convergence_eps':
                self.convergence_eps = float(kwargs[key])
            elif key == 'combined_training_epochs':
                self.combined_training_epochs = int(kwargs[key])
            elif key == 'use_cuda':
                self.use_cuda = bool(kwargs[key])
            elif key == 'profile':
                self.profile = bool(kwargs[key])
            else:
                raise ValueError('Unknown keyword {} in constructor.'.format(key))
        
        # Training parameters
        self.epoch = 0
        
        # training state was filled by previous training or by persisted training state
        if 'trainingstate' in kwargs.keys() and self.trainingstate.state != None:
            self._status_summary()
            self.epoch = self.trainingstate.state["training_loss[]"][-1][0]
            self.trainingstate.load_optimizer_(self.optimizer)
            self.trainingstate.load_weights_(self.model, self.optimizer)
            if isinstance(self.scheduler, StepLR_earlystop):
                self.trainingstate.load_scheduler_(self.scheduler)
        
        # Computational configuration
        if self.use_cuda:
            if not torch.cuda.is_available():
                logger.error('CUDA is not available on your system.')
        self.model = Trainingstate.transform_model(self.model, self.optimizer, 
            self.precision, self.use_cuda)
    
    
    # This depends on the learning problem => needs to be defined by subclass
    def _data_validation(self, dataloader_training, validation_set):
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
        *output    (torch.Tensor): A packed torch.Tensor representing a single output of a mini-batch.
        *targets   (torch.Tensor): The targets for a mini-batch
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        Return
        ------
        *loss      (torch.Tensor): The computed loss as scalar
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        
        """
        if type(output) != tuple and type(output) != list and targets is not None:
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
        *OPTIONAL after_backward_hook (function(output.data, targets.data, loss.data)):
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
            after_backward_hook(output.data, targets.data, loss.cpu().data)
        
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
        # BACKWARD COMPATIBILITY FOR TORCH < 0.4
        if type(value) is not float:
            if type(value) == torch.Tensor:
                value = value.item()
            else:
                value = value.data[0]
        
        n = buffer.shape[0]
        if ma is None:
            moving_average = value
            buffer += value
        else:
            moving_average = ma + (value / n) - (buffer[t % n] / n)
        buffer[t % n] = value
        return moving_average
    
    
    def _input_params_validation(self, epochs):
        """
        Validates the given parameters
        
        Arguments
        ---------
        *epochs (int) : The number of scheduled epochs.
        
        Return
        ------
        *epochs (int) : Same as input or corrected versions of input.
        *eval_interval(int) : Same as input or corrected versions of input.
        """
        # check parameters
        epochs = int(epochs)
        if epochs < 1:
            self.logger.error('Number of epochs must be > 0.', ValueError)
        
        return epochs    
    
    
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
        self.logger.print_problem_summary(self, self.model, self.criterion, self.optimizer, 
            dataloader_training, validation_set, epochs, early_stopping, self.combined_training_epochs)
        
        # training startup message
        self.logger.info('Begin training: {} epochs.'.format(epochs))    
        
        
    def _status_summary(self):
        if self.trainingstate.state is not None:
            self.logger.info("\n[Status]" )
            trs = self.trainingstate.get_summary()
            self.logger.info('Epochs: {}, best training loss ({}): {:.4f}, best validation loss ({}): {:.4f}'.format(
                trs['Epochs'], trs['Best Training Loss'][0], trs['Best Training Loss'][1], 
                trs['Best Validation Loss'][0], trs['Best Validation Loss'][1]))
    
    
    def _evaluate_training(self, Analyzer, batch, batches, 
                  time_dict, 
                  epoch,  
                  validation_set, 
                  avg_training_loss,
                  dataloader_training, 
                  after_eval_hook, 
                  eval_batch_size,
                  output_buffer):
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
        if validation_set is not None:
            
                # MODEL EVALUATION
                evaluation_dict = Analyzer.evaluate(self.model, 
                                                    self.criterion, 
                                                    validation_set, 
                                                    self.logger, 
                                                    after_eval_hook,
                                                    eval_batch_size,
                                                    output_buffer)
                
                # UPDATE TRAININGSTATE
                self.trainingstate.update_state(epoch + 1, self.model, self.criterion, self.optimizer, 
                                        training_loss = avg_training_loss, 
                                        training_accuracy = evaluation_dict["training_accuracy"],
                                        training_batchsize = dataloader_training.batch_size,
                                        training_dataset = dataloader_training.dataset,
                                        trainer_instance = type(self),
                                        precision = self.precision,
                                        detailed_loss = evaluation_dict["detailed_loss"],
                                        validation_loss = evaluation_dict["loss"],
                                        validation_accuracy = evaluation_dict["accuracy"],  
                                        validation_dataset = validation_set,
                                        samples_per_second = evaluation_dict["samples_per_second"],
                                        scheduler = self.scheduler,
                                        combined_retraining = self.combined_training_epochs)
                                
        else: # no validation set
            self.trainingstate.update_state(epoch + 1, self.model, self.criterion, self.optimizer, 
                training_loss = avg_training_loss, 
                training_batchsize = dataloader_training.batch_size,
                training_dataset = dataloader_training.dataset,
                trainer_instance = type(self),
                precision = self.precision,
                detailed_loss = repr(self.criterion))
        
        self.logger.log_epoch(epoch + 1, batch + 1, 
                              batches, 
                              avg_training_loss, 
                              dataloader_training.batch_size, 
                              time_dict,
                              Analyzer.evalstr(self.trainingstate), 
                              self.profile,
                              evaluation_dict)
    
    
    def _combined_retraining(self, dataloader_training, validation_set, 
                             after_backward_hook, after_eval_hook, eval_batch_size):
        """
        Does combined retraining with validation AND training data. Can be used after the normal training to refine the model.
        
        Arguments
        ---------
        dataloader_training :   torch.utils.data.DataLoader OR tuple (X, batch)
                                The dataloader that provides the training data
        validation_set      :   torch.utils.data.Dataset OR tuple (X)
                                The validation dataset
        after_backward_hook :   OPTIONAL function(output.data, targets.data, loss.data)
                                A hook that gets called after backward pass during training
        after_eval_hook     :   OPTIONAL function(output.data, targets.data, loss.data)
                                A hook that gets called after forward pass during evaluation
        eval_batch_size     :   OPTIONAL int
                                batch size used for evaluation (default: -1 == ALL)
        
        """
        if self.combined_training_epochs > 0:
            if validation_set is None:
                self.logger.warn("Combined retraining needs validation data.")
            else:                
                # combine the two datasets
                dataloader_combined = self._add_dataset_to_loader(dataloader_training, validation_set)   
                
                # give some information about what we are going to do
                self.logger.info('Combined retraining...')  
                
                # get current state
                combined_training_epochs = self.combined_training_epochs
                model_filename = self.model_filename
                model_keep_epochs = self.model_keep_epochs
                self.scheduler = None
                
                # modify state so that recursion is not infinite, and filenames are correct
                self.combined_training_epochs = 0
                self.model_filename = str(self.model_filename + self.trainingstate.combined_retraining_pattern)
                self.model_keep_epochs = True
                
                # reset to best validation model
                self.trainingstate.load_weights_best_validation_(self.model, self.optimizer)
                
                # do actual retraining
                self.fit(dataloader_combined, 
                         epochs=combined_training_epochs, 
                         validation_set=validation_set, 
                         after_backward_hook=after_backward_hook, 
                         after_eval_hook=after_eval_hook, 
                         eval_batch_size=eval_batch_size)
                
                # restore previous state
                self.combined_training_epochs = combined_training_epochs
                self.model_filename = model_filename
                self.model_keep_epochs = model_keep_epochs
    
    def _add_dataset_to_loader(self, dataloader, merge_dataset):
        """
        Adds a dataset to an existing dataloader
        
        dataloader (torch.utils.data.DataLoader) : A new instance of a dataloader that contains the merged dataset
        """
        dataset_origin = dataloader.dataset
        dataset_merged = ConcatDataset([dataset_origin, merge_dataset])
        dataloader_merged = DataLoader(dataset_merged, 
                                       batch_size=dataloader.batch_size, 
                                       shuffle=True, 
                                       num_workers=dataloader.num_workers)
        return dataloader_merged
        
    # prepares one batch for processing (can be overwritten by sibling)
    def _get_batch(self, data):
        
        # Get the inputs
        inputs, targets = Variable(data[0]), Variable(data[1])
        
        # Handle cuda
        if self.use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        
        return inputs, targets
    
    
    def fit(self, dataloader_training, epochs=1, validation_set=None, 
        after_backward_hook=None, after_eval_hook=None, eval_batch_size=-1):
        """
        Fits a model with given training and validation dataset
        
        Arguments
        ---------
        dataloader_training :   torch.utils.data.DataLoader OR tuple (X,y,batch)
                                The dataloader that provides the training data
        epochs              :   int
                                Epochs to train
        validation_set      :   torch.utils.data.Dataset OR tuple (X,y)
                                The validation dataset
        eval_interval       :   int
                                Evaluation interval for validation dataset in epochs
        after_backward_hook :   OPTIONAL function(output.data, targets.data, loss.data)
                                A hook that gets called after backward pass during training
        after_eval_hook     :   OPTIONAL function(ctx, output.data, targets.data, loss.data)
                                A hook that gets called after forward pass during evaluation
        eval_batch_size     :   OPTIONAL int
                                batch size used for evaluation (default: -1 == ALL)
        """
        
        # INPUT VALIDATION
        dataloader_training, validation_set, batches = self._data_validation(
            dataloader_training, validation_set)
        epochs = self._input_params_validation(epochs)
        if eval_batch_size == -1:
            eval_batch_size = dataloader_training.batch_size

        # PROBLEM SUMMARY
        self._problem_summary(epochs, dataloader_training, validation_set, self.scheduler)
        
        for epoch in range(self.epoch, self.epoch + epochs):
        
            # EPOCH PREPARATION
            time_dict = self._prepare_epoch()
            
            # Moving average
            n, avg_training_loss = 5, None
            training_loss_buffer= np.zeros(n, dtype=np.float32)
            
            # Buffer for asynchronous model evaluation
            output_buffer = []
            
            # COMPUTE ONE EPOCH                
            for batch, data in enumerate(dataloader_training, 0):
                
                # time dataloader
                time_dict["loader"] = time_dict["loader"] + (time.time() - time_dict["t"])
                
                # Get the inputs
                inputs, targets = self._get_batch(data)
                
                output, time_dict = self._forward_one_batch(inputs, time_dict)
                loss,   time_dict = self._loss_one_batch(output, targets, time_dict)

                # Backpropagation
                time_dict = self._backward_one_batch(loss, time_dict, after_backward_hook, 
                    output, targets)
                
                # Loss averaging
                avg_training_loss = self._moving_average(batch, avg_training_loss, 
                    loss.cpu(), training_loss_buffer)
                
                # Reporting
                time_dict = self._finish_one_batch(batch, batches, epoch, avg_training_loss,
                    dataloader_training.batch_size, time_dict)
                
                # Save output for later evaluation
                output_buffer.append((output.data.clone(), targets.data.clone(), batch))
            
            # Evaluate
            self._evaluate_training(self.analyzer, batch, batches, time_dict, epoch,  
                validation_set, avg_training_loss, dataloader_training, after_eval_hook, 
                eval_batch_size, output_buffer)
            
            # SAVE MODEL
            self.trainingstate.save_state(self.model_filename, self.model_keep_epochs)
                     
            # CHECK TRAINING CONVERGENCE
            if self._has_converged():
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
        
        
        
        
