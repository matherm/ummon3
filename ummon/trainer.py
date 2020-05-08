import time
import numpy as np
import types
import torch.nn as nn
import torch.utils.data
import ummon.utils as uu
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from torch.utils.data.dataset import Subset
from .logger import Logger
from .schedulers import *
from .trainingstate import *
from .analyzer import Analyzer

__all__ = ["Trainer", "SupervisedTrainer", "ClassificationTrainer", "UnsupervisedTrainer", "SiameseTrainer", "KamikazeTrainer"]

class Trainer:
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
    optim_closure     : callable
                        OPTIONAL Callable closure that gets passed to step(closure) of optimizer if torch.optim.LBFGS                      
    
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
        if self.criterion is None:
            self.criterion = lambda X,y : X  # just replays model output
        assert isinstance(self.criterion, nn.Module) or isinstance(self.criterion, types.LambdaType)
        self.optimizer = args[3]
        if self.optimizer is None:
            self.optimizer = torch.optim.SGD([Variable(torch.zeros(0))], lr=0) # just does nothing
        assert isinstance(self.optimizer, torch.optim.Optimizer)
        
        # defaults
        self.trainingstate = Trainingstate(filename=None, model_keep_epochs=False)
        self.scheduler = None
        self.precision = np.float32
        self.convergence_eps = np.finfo(np.float32).min
        self.combined_training_epochs = False
        self.use_cuda = False
        
        # optional arguments
        for key in kwargs:
            if key == 'trainingstate':
                if not isinstance(kwargs[key], Trainingstate):
                    raise TypeError('{} is not a training state'.format(type(kwargs[key]).__name__))
                self.trainingstate = kwargs[key]
                if 'model_filename' in kwargs.keys():
                    self.trainingstate.filename = str(kwargs["model_filename"]).split(self.trainingstate.extension)[0]            
                if 'model_keep_epochs' in kwargs.keys():     
                    self.trainingstate.model_keep_epochs = int(kwargs["model_keep_epochs"])
            elif key == 'scheduler':
                if not isinstance(kwargs[key], torch.optim.lr_scheduler._LRScheduler) and not \
                    isinstance(kwargs[key], StepLR_earlystop):
                    raise TypeError('{} is not a scheduler'.format(type(kwargs[key]).__name__))
                if isinstance(kwargs[key], StepLR_earlystop) and 'trainingstate' not in kwargs.keys():
                    raise ValueError('StepLR_earlystop needs an external Trainingstate (you provided None).')
                self.scheduler = kwargs[key]
            elif key == 'precision':
                assert kwargs[key] == np.float32 or kwargs[key] == np.float64
                self.precision = kwargs[key]
            elif key == 'convergence_eps':
                self.convergence_eps = float(kwargs[key])
            elif key == 'combined_training_epochs':
                if self.trainingstate.filename is None:
                    raise ValueError('Combined retraining needs a model_filename to load the best model after training. (you provided None).')
                self.combined_training_epochs = int(kwargs[key])
            elif key == 'model_keep_epochs':
                self.trainingstate.model_keep_epochs = int(kwargs[key])
            elif key == 'model_filename':
                self.trainingstate.filename = str(kwargs[key]).split(self.trainingstate.extension)[0]            
            elif key == 'use_cuda':
                self.use_cuda = kwargs[key]
            elif key == 'optim_closure':
                self.optim_closure = kwargs[key]
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
                self.logger.error('CUDA is not available on your system.')
        else:
            self.use_cuda = next(self.model.parameters()).is_cuda
        self.model = Trainingstate.transform_model(self.model, self.optimizer, 
            self.precision, self.use_cuda)


    def fit(self, dataloader_training, epochs=1, validation_set=None, eval_interval=1, eval_batch_size=-1, analyzer=Analyzer, metrics=[]):
        """
        Fits a model with given training and validation dataset
        
        Arguments
        ---------
        dataloader_training :   torch.utils.data.DataLoader OR tuple (X,y,batch)
                                The dataloader that provides the training data
        epochs              :   int
                                Epochs to train
        validation_set      :   torch.utils.data.Dataset OR tuple (X,y) OR torch.utils.data.DataLoader
                                The validation dataset
        eval_interval       :   int
                                Evaluation interval for validation dataset in epochs
        eval_batch_size     :   OPTIONAL int
                                batch size used for evaluation (default: -1 == ALL)
        Analyzer (ummon.Analyzer) : A training type specific analyzer.
        metrics (ummon.metrics) : a list of metrics
        """
        
        # INPUT VALIDATION
        epochs = int(epochs)
        if epochs < 1:
            self.logger.error('Number of epochs must be > 0.', ValueError)
        dataloader_training = uu.gen_dataloader(dataloader_training, has_labels=True, logger=self.logger)
        batches = len(dataloader_training)
        
        if eval_batch_size == -1:
            eval_batch_size = dataloader_training.batch_size
        
        dataloader_validation = uu.gen_dataloader(validation_set, batch_size=eval_batch_size, has_labels=True, logger=self.logger)
        
        # PROBLEM SUMMARY
        self._problem_summary(epochs, dataloader_training, dataloader_validation, self.scheduler)
        
        for epoch in range(self.epoch, self.epoch + epochs):

            # switch on training mode
            self.model.train()
        
            # EPOCH PREPARATION
            time_dict = self._prepare_epoch()
            
            # COMPUTE ONE EPOCH                
            for batch, data in enumerate(dataloader_training, 0):
                
                # time dataloader
                time_dict["loader"] = time_dict["loader"] + (time.time() - time_dict["t"])
                
                # Get the inputs
                inputs, targets = self._get_batch(data)                
                
                output, time_dict = self._forward_one_batch(inputs, time_dict)
                loss,   time_dict = self._loss_one_batch(output, targets, time_dict)
                
                # Backpropagation
                time_dict = self._backward_one_batch(loss, time_dict, output, targets)
                
                # Reporting
                time_dict = self._finish_one_batch(batch, batches, epoch, loss.item(),
                    dataloader_training.batch_size, time_dict)
                
            # Evaluate
            if epoch % eval_interval == 0 and not torch.isnan(loss):
                
                self.model.eval()
                
                self._evaluate_training(analyzer, batch, batches, time_dict, epoch,  
                    dataloader_validation, dataloader_training, eval_batch_size, metrics)
                
                # SAVE MODEL
                self.trainingstate.save_state()
                     
                # CHECK TRAINING CONVERGENCE
                if self._has_converged():
                    break

            if torch.isnan(loss) and epoch == 0:
                raise Exception("Loss was NaN in 1st epoch.")
            
            # ANNEAL LEARNING RATE
            if self.scheduler: 
                try:
                    self.scheduler.step()
                except StepsFinished:
                    break
                
        # DO COMBINED RETRAINING WITH BEST VALIDATION MODEL
        self._combined_retraining(dataloader_training, validation_set, eval_batch_size) 
    
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
        if type(output) != tuple and type(output) != list and targets is not None and type(targets) != list and type(targets) != tuple:
                if targets.is_cuda or output.is_cuda:
                    output, targets = output.to('cuda'), targets.to('cuda')
                    
        if (type(output) == tuple or type(output) == list) and targets is not None:
            if output[0].is_cuda or output[1].is_cuda:
                output = uu.tensor_tuple_to_cuda(output)
                if type(targets) == tuple or type(targets) == list:
                    targets = uu.tensor_tuple_to_cuda(targets)
        
        try:
            loss = self.criterion(output, targets)
        except (ValueError, TypeError):
            try:
                # case: loss does not have targets e.g. entropy loss
                loss = self.criterion(output)
            except (ValueError, TypeError):
                    try:
                        # case: case targets are not formatted correctly
                        loss = self.criterion(output, targets.view_as(output))
                    except RuntimeError:
                        # case: targets are not formatted correctly
                        loss = self.criterion(output, targets.view_as(output).float())
        
        # time loss
        time_dict["loss"] = time_dict["loss"] + (time.time() - time_dict["t"])
        return loss, time_dict
      
    def _backward_one_batch(self, loss, time_dict, output=None, targets=None):
        """
        Computes the loss for a single mini-batch
        
        Arguments
        ---------
        *loss      (torch.autograd.Variable): The computed loss as scalar
        *time_dict (dict)                   : Dictionary that is used for profiling executing time.
        *OPTIONAL output    (torch.autograd.Variable): A packed torch.Tensor representing a single output of a mini-batch.
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
        time_dict["backprop"] = time_dict["backprop"] + (time.time() - time_dict["t"])
        
        # Take gradient descent
        if hasattr(self, "optim_closure"):
            self.optimizer.step(self.optim_closure)
        else:
            self.optimizer.step()
        
        return time_dict

    def _finish_one_batch(self, batch, batches, epoch, training_loss, training_batchsize, time_dict):
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
        time_dict["total"] = time_dict["total"] + (time.time() - time_dict["t"])

        # Log status
        self.logger.log_one_batch(epoch + 1, batch + 1, batches, training_loss, training_batchsize, time_dict)
        
        # Reset time
        time_dict["t"] = time.time()
        return time_dict
    
    
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

    
    def _problem_summary(self, epochs, dataloader_training, dataloader_validation, scheduler = None):
        """
        Prints the problem summary
        
        Arguments
        ---------
        *epochs (int) : The number of scheduled epochs.
        *dataloader_training (torch.utils.data.Dataloader) : A dataloader holding the training data.
        *dataloader_validation (torch.utils.data.Dataloader) : A dataset holding the validation data
        """
        # PRINT SOME INFORMATION ABOUT THE SCHEDULED TRAINING
        early_stopping = isinstance(scheduler, StepLR_earlystop)
        validation_dataset = dataloader_validation.dataset if dataloader_validation is not None else None
        self.logger.print_problem_summary(self, self.model, self.criterion, self.optimizer, 
            dataloader_training.dataset, dataloader_training.batch_size, validation_dataset, epochs, early_stopping, self.combined_training_epochs)
        
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
                  dataloader_validation, 
                  dataloader_training,
                  eval_batch_size,
                  metrics):
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
        *dataloader_validation (torch.utils.data.Dataloader) : Validation data.
        *dataloader_training : The training data
        *eval_batch_size (int) : A custom batch size to be used during evaluation
        *metrics (list) : a list of ummon.metrics
        
        """
        # EVALUATE ON TRAINING SET
        validation_loss = None
        evaluation_dict_train = Analyzer.evaluate(self.model, 
                                                self.criterion, 
                                                dataloader_training,
                                                batch_size=eval_batch_size,
                                                limit_batches=200,
                                                logger=self.logger,
                                                metrics=metrics)
        
        # Log epoch
        if dataloader_validation is not None:
            
                # MODEL EVALUATION
                evaluation_dict = Analyzer.evaluate(self.model, 
                                                    self.criterion, 
                                                    dataloader_validation, 
                                                    batch_size=eval_batch_size,
                                                    limit_batches=-1,
                                                    logger=self.logger,
                                                    metrics=metrics)
        
                # UPDATE TRAININGSTATE
                self.trainingstate.update_state(epoch + 1, self.model, self.criterion, self.optimizer, 
                                        training_loss = evaluation_dict_train["loss"], 
                                        training_accuracy = evaluation_dict_train["accuracy"],
                                        training_dataset = dataloader_training.dataset,
                                        training_batchsize = dataloader_training.batch_size,
                                        trainer_instance = type(self),
                                        precision = self.precision,
                                        detailed_loss = evaluation_dict["detailed_loss"],
                                        validation_loss = evaluation_dict["loss"],
                                        validation_accuracy = evaluation_dict["accuracy"],  
                                        validation_dataset = dataloader_validation.dataset,
                                        samples_per_second = evaluation_dict["samples_per_second"],
                                        scheduler = self.scheduler,
                                        combined_retraining = self.combined_training_epochs,
                                        evaluation_dict_train=evaluation_dict_train,
                                        evaluation_dict_eval=evaluation_dict)
                validation_loss = evaluation_dict["loss"]
                                
        else: # no validation set
            
            evaluation_dict = evaluation_dict_train
            
            self.trainingstate.update_state(epoch + 1, self.model, self.criterion, self.optimizer, 
                training_loss = evaluation_dict_train["loss"], 
                training_accuracy = evaluation_dict_train["accuracy"],
                training_batchsize = dataloader_training.batch_size,
                training_dataset = dataloader_training.dataset,
                trainer_instance = type(self),
                precision = self.precision,
                detailed_loss = repr(self.criterion),
                scheduler = self.scheduler,
                evaluation_dict_train=evaluation_dict_train)

        
        self.logger.log_epoch(epoch + 1, batch + 1, 
                              batches, 
                              dataloader_training.batch_size, 
                              time_dict,
                              self.trainingstate)


        return validation_loss
    
    
    def _combined_retraining(self, dataloader_training, validation_set, eval_batch_size):
        """
        Does combined retraining with validation AND training data. Can be used after the normal training to refine the model.
        
        Arguments
        ---------
        dataloader_training :   torch.utils.data.DataLoader OR tuple (X, batch)
                                The dataloader that provides the training data
        validation_set      :   torch.utils.data.Dataset OR tuple (X)
                                The validation dataset
        eval_batch_size     :   OPTIONAL int
                                batch size used for evaluation (default: -1 == ALL)
        
        """
        if self.combined_training_epochs > 0:
            if validation_set is None:
                self.logger.warn("Combined retraining needs validation data.")
            else:                
                # combine the two datasets
                uu.add_dataset_to_loader_(dataloader_training, validation_set)   
                
                # give some information about what we are going to do
                self.logger.info('Combined retraining...')  
                
                # get current state
                combined_training_epochs = self.combined_training_epochs
                self.scheduler = None
                
                # reset to best validation model
                self.trainingstate.load_weights_best_validation_(self.model, self.optimizer)
                
                # modify state so that recursion is not infinite, and filenames are correct
                self.combined_training_epochs = 0
                self.trainingstate.add_combined_retraining_pattern()
                
                # do actual retraining
                self.fit(dataloader_training, 
                         epochs=combined_training_epochs, 
                         validation_set=validation_set, 
                         eval_batch_size=eval_batch_size)
                
                # restore previous state
                self.combined_training_epochs = combined_training_epochs
                self.trainingstate.remove_combined_retraining_pattern()

        
    # prepares one batch for processing (can be overwritten by sibling)
    def _get_batch(self, data):

        inputs, targets = uu.input_of(data), uu.label_of(data)
        
        # Get the inputs
        inputs, targets = uu.tensor_tuple_to_variables(inputs), uu.tensor_tuple_to_variables(targets)
        
        # Handle cuda
        if self.use_cuda:
            inputs, targets = uu.tensor_tuple_to_cuda(inputs), uu.tensor_tuple_to_cuda(targets)
        
        return inputs, targets
        
        
# Backward compatibility
class SupervisedTrainer(Trainer):
    pass

# Backward compatibility
class UnsupervisedTrainer(Trainer):
    pass

# Backward compatibility
class SiameseTrainer(Trainer):
    pass

from .analyzer import ClassificationAnalyzer
class ClassificationTrainer(Trainer):
    
    def fit(self, dataloader_training, epochs=1, validation_set=None, eval_interval=1, eval_batch_size=-1):
        return super().fit(dataloader_training, epochs, validation_set, eval_interval, eval_batch_size, ClassificationAnalyzer)

class KamikazeTrainer(Trainer):
    """
    The most simple trainer that simply pushes batches directly from the loader into the model.
    This handling also applies to model outputs the loss function.

    Usage:
        X = np.zeroes(2000,20)    
        KamikazeTrainer().fit(model, X)
    """

    def _get_batch(self, data):
        
        if type(data) == list:
            inputs, targets = data
            if self.use_cuda:
                inputs, targets = inputs.to('cuda'), targets.to('cuda')
            return inputs, targets
        else:
            if self.use_cuda:
                if hasattr(data, "cuda"):
                    data = data.to('cuda')
                elif hasattr(data, "to"):
                    data = data.to("cuda")
            return data, data

    def _loss_one_batch(self, output, targets, time_dict):
        
        loss = self.criterion(output, targets)

        time_dict["loss"] = time_dict["loss"] + (time.time() - time_dict["t"])

        return loss, time_dict
