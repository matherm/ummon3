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
from ummon.utils import Torchutils
from ummon.logger import Logger
from ummon.trainingstate import Trainingstate
from ummon.analyzer import Analyzer

class Trainer:
    """
    This class provides a generic trainer for training PyTorch-models.
    
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
    regression        : bool
                        OPTIONAL Speficies the problem type. Used for outputs etc (default False).
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
                 regression = False, 
                 model_filename = "model.pth.tar", 
                 model_keep_epochs = False,
                 precision = np.float32,
                 use_cuda = False,
                 profile = False):
        
        assert type(logger) == Logger
        assert isinstance(model, nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) if not scheduler is None else True
        assert precision == np.float32 or precision == np.float64
        
        self.name = "ummon.Trainer"

        # MEMBER VARIABLES
        # Training parameters
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.regression = regression
        self.epoch = 0
        self.precision = precision
        self.use_cuda = use_cuda
        self.profile = profile
        
        #  Persistency parameters
        self.model_filename = model_filename
        self.model_keep_epochs = model_keep_epochs
        
        # INITIALIZE LOGGER
        self.logger = logger
        
        # Computational configuration
        if self.use_cuda:
            if not torch.cuda.is_available():
                logger.error('CUDA is not available on your system.')
        self.model = Torchutils.transform_model(model, precision, use_cuda)
    
    
    def fit(self, dataloader_training, epochs=1, validation_set=None, eval_interval=500, 
        trainingstate=None, after_backward_hook=None, after_eval_hook=None, 
        eval_batch_size=-1, args=None):
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
        trainingstate       :   ummon.Trainingstate
                                OPTIONAL An optional trainingstate to initialize model and optimizer with an previous state
        after_backward_hook :   OPTIONAL function(model, output.data, targets.data, loss.data)
                                A hook that gets called after backward pass during training
        after_eval_hook     :   OPTIONAL function(model, output.data, targets.data, loss.data)
                                A hook that gets called after forward pass during evaluation
        eval_batch_size     :   OPTIONAL int
                                batch size used for evaluation (default: -1 == ALL)
        args                :   OPTIONAL dict
                                A dict that gets persisted and can hold arbitrary information about the trainine
        
        Return
        ------
        ummon.Trainingstate
        A dictionary containing the trainingstate
        """
        assert type(trainingstate) == Trainingstate if not trainingstate is None else True
               
        # RESTORE STATE    
        if trainingstate:
            self.logger.info("[Status]" )
            trs = trainingstate.get_summary()
            self.logger.info('Epochs: {}, best training loss ({}): {:.4f}, best validation loss ({}): {:.4f}'.format(
                trs['Epochs'], trs['Best Training Loss'][0], trs['Best Training Loss'][1], 
                trs['Best Validation Loss'][0], trs['Best Validation Loss'][1]))
            self.epoch = trainingstate.state["training_loss[]"][-1][0]
            self.model = trainingstate.load_weights(self.model, self.precision, self.use_cuda)
            self.optimizer = trainingstate.load_optimizer(self.optimizer)
            
        else:
            trainingstate = Trainingstate()
            
        # simple interface: training and test data given as numpy arrays
        if type(dataloader_training) == tuple:
            dataset = Torchutils.construct_dataset_from_tuple(logger=self.logger, data_tuple=dataloader_training, train=True)
            batch = int(dataloader_training[2])
            dataloader_training = DataLoader(dataset, batch_size=batch, shuffle=True, 
                sampler=None, batch_sampler=None)
        assert isinstance(dataloader_training, torch.utils.data.DataLoader)
        assert Torchutils.check_precision(dataloader_training.dataset, self.model, self.precision)
        if validation_set is not None:
            if type(validation_set) == tuple:
                validation_set = Torchutils.construct_dataset_from_tuple(logger=self.logger, data_tuple=validation_set, train=False)
            assert isinstance(validation_set, torch.utils.data.Dataset)
            assert Torchutils.check_precision(validation_set, self.model, self.precision)
        
        # check parameters
        epochs = int(epochs)
        if epochs < 1:
            self.logger.error('Number of epochs must be > 0.', ValueError)
        eval_interval = int(eval_interval)
        
        # COMPUTE BATCHES PER EPOCH
        batches = int(np.ceil(len(dataloader_training.dataset) / dataloader_training.batch_size))
        
        # PRINT SOME INFORMATION ABOUT THE SCHEDULED TRAINING
        self.logger.print_problem_summary(self.model, self.criterion, self.optimizer, 
            dataloader_training, validation_set, epochs, None)
        
        # training startup message
        self.logger.info('Begin training: {} epochs.'.format(epochs))        
        for epoch in range(self.epoch, self.epoch + epochs):
            
            # TAKE TIME
            t = time.time()
            time_dict = {"loader"   : 0,
                         "model"    : 0,
                         "loss"     : 0,
                         "backprop" : 0,
                         "hooks"    : 0,
                         "total"    : 0}
            
            # ANNEAL LEARNING RATE
            if self.scheduler: 
                self.scheduler.step()
           
            # Moving average
            n, avg_training_loss, avg_training_acc = 5, None, None
            training_loss, training_acc  = np.zeros(n, dtype=np.float64), np.zeros(n, 
                dtype=np.float64)
            
            # Buffer for asynchronous model evaluation
            output_buffer = []
            
            # COMPUTE ONE EPOCH                
            for batch, data in enumerate(dataloader_training, 0):
                
                # time dataloader
                time_dict["loader"] = time_dict["loader"] + (time.time() - t)
                
                # Get the inputs
                inputs, targets = Variable(data[0]), Variable(data[1])
                
                # Handle cuda
                if self.use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                
                # Execute Model
                output = self.model(inputs)
                
                # time model
                if self.profile and self.use_cuda: torch.cuda.synchronize()
                time_dict["model"] = time_dict["model"] + (time.time() - t)
                
                # Ensure Cuda for singular outputs
                if type(output) != tuple:
                    if targets.is_cuda or output.is_cuda:
                        output, targets = output.cuda(), targets.cuda()
                    else:
                        output, targets = output.cpu(), targets.cpu()
                    assert type(output) == torch.autograd.Variable
                else:
                    assert type(output[0]) == torch.autograd.Variable
                
                # Compute Loss
                loss = self.criterion(output, targets)
                
                # time loss
                if self.profile and self.use_cuda: torch.cuda.synchronize()
                time_dict["loss"] = time_dict["loss"] + (time.time() - t)
                
                # Zero the gradient    
                self.optimizer.zero_grad()
                
                # Backpropagation
                loss.backward()
                
                # time backprop
                if self.profile and self.use_cuda: torch.cuda.synchronize()
                time_dict["backprop"] = time_dict["backprop"] + (time.time() - t)
                
                # Run hooks
                if after_backward_hook is not None:
                    if type(output) != tuple:
                        after_backward_hook(self.model, output.data, targets.data, loss.cpu().data)
                    else:
                        after_backward_hook(self.model, output[0].data, targets.data, loss.cpu().data)
                
                # time hooks
                if self.profile and self.use_cuda: torch.cuda.synchronize()
                time_dict["hooks"] = time_dict["hooks"] + (time.time() - t)
                
                # Take gradient descent
                self.optimizer.step()
                
                # Running average training loss
                avg_training_loss = self._moving_average(batch, avg_training_loss, loss.cpu().data[0], training_loss)
                
                # total time
                if self.profile and self.use_cuda: torch.cuda.synchronize()
                time_dict["total"] = time_dict["total"] + (time.time() - t)
                
                # Log status
                self.logger.log_one_batch(epoch + 1, batch + 1, batches, avg_training_loss, dataloader_training.batch_size, time_dict, self.profile)
                
                # Reset time
                t = time.time()
                
                # Save output for later evaluation
                if not self.regression:
                    if type(output) != tuple:
                        output_buffer.append((output.data.clone(), targets.data.clone(), batch))
                    else:
                        output_buffer.append((output[0].data.clone(), targets.data.clone(), batch))
                else:
                    avg_training_acc = 0.
            
            # Log epoch
            self.logger.log_epoch(epoch + 1, batch + 1, 
                                  batches, 
                                  avg_training_loss, 
                                  dataloader_training.batch_size, 
                                  time_dict, 
                                  self.profile)
            
            # MODEL VALIDATION
            trainingstate = self._evaluate(epoch + 1, eval_interval, validation_set, avg_training_loss, 
                                               avg_training_acc, training_acc, dataloader_training.batch_size, 
                                               dataloader_training, after_eval_hook, eval_batch_size, 
                                               trainingstate, output_buffer, args)
                    
            # SAVE MODEL
            trainingstate.save_state(self.model_filename, self.model_keep_epochs)
            
            # CLEAN UP
            del output_buffer[:]
                
        return trainingstate
    
    
    def _evaluate(self, epoch, eval_interval, validation_set, avg_training_loss, avg_training_acc, training_acc,
        training_batch_size, dataloader_training, after_eval_hook, eval_batch_size, 
        trainingstate, output_buffer, args):
        # INIT ARGS
        args = {} if args is None else args
        
        if (epoch +1) % eval_interval == 0 and validation_set is not None:
                # Compute Running average accuracy
                for saved_output, saved_targets, batch in output_buffer:
                    if type(saved_output) != tuple:
                        classes = Analyzer.classify(saved_output.cpu())
                    else:
                        classes = Analyzer.classify(saved_output[0].cpu())
                    acc = Analyzer.compute_accuracy(classes, saved_targets.cpu())
                    avg_training_acc = self._moving_average(batch, avg_training_acc, acc, training_acc)
        
            
                # MODEL EVALUATION
                evaluation_dict = Analyzer.evaluate(self.model, 
                                                    self.criterion, 
                                                    validation_set, 
                                                    self.regression, 
                                                    self.logger, 
                                                    after_eval_hook, 
                                                    batch_size=eval_batch_size)
                
                # UPDATE TRAININGSTATE
                trainingstate.update_state(epoch, self.model, self.criterion, self.optimizer, 
                             training_loss = avg_training_loss, 
                             validation_loss = evaluation_dict["loss"], 
                             training_accuracy = avg_training_acc,
                             training_batchsize = training_batch_size,
                             validation_accuracy = evaluation_dict["accuracy"], 
                             validation_batchsize = len(validation_set),
                             regression = self.regression,
                             precision = self.precision,
                             detailed_loss = evaluation_dict["detailed_loss"],
                             training_dataset = dataloader_training.dataset,
                             validation_dataset = validation_set,
                             samples_per_second = evaluation_dict["samples_per_second"],
                             args = {**args, **evaluation_dict["args[]"]})

                self.logger.log_evaluation(trainingstate, self.profile)
                        
        else: # no validation set
                    
                    trainingstate.update_state(epoch, self.model, self.criterion, self.optimizer, 
                        training_loss = avg_training_loss, 
                        validation_loss = None, 
                        training_accuracy = avg_training_acc,
                        training_batchsize = dataloader_training.batch_size,
                        validation_accuracy = None, 
                        validation_batchsize = None,
                        regression = self.regression,
                        precision = self.precision,
                        detailed_loss = repr(self.criterion),
                        training_dataset = dataloader_training.dataset,
                        validation_dataset = None,
                        samples_per_second = None,
                        args = {})

        return trainingstate
        
        
    def _moving_average(self, t, ma, value, buffer):
        n = buffer.shape[0]
        if ma is None:
            moving_average = value
            buffer += value
        else:
            moving_average = ma + (value / n) - (buffer[t % n] / n)
        buffer[t % n] = value
        return moving_average
               
        
if __name__ == "__main__":
    print("This is", Trainer().name)
    
