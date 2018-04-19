import time
import numpy as np
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from torch.utils.data import DataLoader
from ummon import *

class MetaTrainer:
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
        
        assert type(logger) == Logger
        assert isinstance(model, nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) if not scheduler is None else True
        assert precision == np.float32 or precision == np.float64
        
        # MEMBER VARIABLES
        # Training parameters
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
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
        self.model = uu.transform_model(model, precision, use_cuda)
    
    
    def _prepare_epoch(self):     
        # TAKE TIME
        time_dict = {"t"        : time.time(),
                     "loader"   : 0,
                     "model"    : 0,
                     "loss"     : 0,
                     "backprop" : 0,
                     "hooks"    : 0,
                     "total"    : 0}
        
        # ANNEAL LEARNING RATE
        if self.scheduler: 
            self.scheduler.step()
        
        return time_dict

        
    def _forward_one_batch(self, inputs, time_dict):
        # Execute Model
        output = self.model(inputs)
        
        # time model
        if self.profile and self.use_cuda: torch.cuda.synchronize()
        time_dict["model"] = time_dict["model"] + (time.time() - time_dict["t"])
       
        return output, time_dict
       
    def _loss_one_batch(self, output, targets, time_dict):

        assert type(output) == torch.autograd.Variable

        if targets.is_cuda or output.is_cuda:
            output, targets = output.cuda(), targets.cuda()
        
        loss = self.criterion(output, targets)
        
        # time loss
        if self.profile and self.use_cuda: torch.cuda.synchronize()
        time_dict["loss"] = time_dict["loss"] + (time.time() - time_dict["t"])
        return loss, time_dict
      
    def _backward_one_batch(self, loss, after_backward_hook, output, targets, time_dict):

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

    def _finish_one_batch(self, batch, batches, epoch, avg_training_loss, current_loss, training_loss_buffer, training_batchsize, time_dict):
        
        # Running average training loss
        avg_training_loss = self._moving_average(batch, avg_training_loss, current_loss.cpu().data[0], training_loss_buffer)
        
        # total time
        if self.profile and self.use_cuda: torch.cuda.synchronize()
        time_dict["total"] = time_dict["total"] + (time.time() - time_dict["t"])
        
        # Log status
        self.logger.log_one_batch(epoch + 1, batch + 1, batches, avg_training_loss, training_batchsize, time_dict, self.profile)
        
        # Reset time
        time_dict["t"] = time.time()
        return avg_training_loss, time_dict
        
        
    def _moving_average(self, t, ma, value, buffer):
        n = buffer.shape[0]
        if ma is None:
            moving_average = value
            buffer += value
        else:
            moving_average = ma + (value / n) - (buffer[t % n] / n)
        buffer[t % n] = value
        return moving_average
    
    
    def _restore_training_state(self, trainingstate):
        assert type(trainingstate) == Trainingstate if not trainingstate is None else True
               
        # RESTORE STATE    
        if trainingstate:
            self.logger.info("[Status]" )
            trs = trainingstate.get_summary()
            self.logger.info('Epochs: {}, best training loss ({}): {:.4f}, best validation loss ({}): {:.4f}'.format(
                trs['Epochs'], trs['Best Training Loss'][0], trs['Best Training Loss'][1], 
                trs['Best Validation Loss'][0], trs['Best Validation Loss'][1]))
            self.epoch = trainingstate.state["training_loss[]"][-1][0]
            self.model = trainingstate.load_weights(self.model)
            self.optimizer = trainingstate.load_optimizer(self.optimizer)
            
        else:
            trainingstate = Trainingstate()
        return trainingstate
            
    
    def _input_data_validation(self, dataloader_training, validation_set):
        # simple interface: training and test data given as numpy arrays
        if type(dataloader_training) == tuple:
            dataset = uu.construct_dataset_from_tuple(logger=self.logger, data_tuple=dataloader_training, train=True)
            batch = int(dataloader_training[2])
            dataloader_training = DataLoader(dataset, batch_size=batch, shuffle=True, 
                sampler=None, batch_sampler=None)
        assert isinstance(dataloader_training, torch.utils.data.DataLoader)
        assert uu.check_precision(dataloader_training.dataset, self.model, self.precision)
        if validation_set is not None:
            if type(validation_set) == tuple:
                validation_set = uu.construct_dataset_from_tuple(logger=self.logger, data_tuple=validation_set, train=False)
            assert isinstance(validation_set, torch.utils.data.Dataset)
            assert uu.check_precision(validation_set, self.model, self.precision)
            
        # COMPUTE BATCHES PER EPOCH
        batches = int(np.ceil(len(dataloader_training.dataset) / dataloader_training.batch_size))
       
        return dataloader_training, validation_set, batches
    
    def _input_params_validation(self, epochs, eval_interval):
        # check parameters
        epochs = int(epochs)
        if epochs < 1:
            self.logger.error('Number of epochs must be > 0.', ValueError)
        eval_interval = int(eval_interval)
        
        return epochs, eval_interval     
    
    def _problem_summary(self, epochs, dataloader_training, validation_set):
        # PRINT SOME INFORMATION ABOUT THE SCHEDULED TRAINING
        self.logger.print_problem_summary(self.model, self.criterion, self.optimizer, 
            dataloader_training, validation_set, epochs, None)
        
        # training startup message
        self.logger.info('Begin training: {} epochs.'.format(epochs))    

        