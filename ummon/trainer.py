#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################

import time
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
from ummon.logger import Logger2
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
    trainingstate     : ummon.Trainingstate
                        OPTIONAL An optional trainingstate to initialize model and optimizer with an previous state
    regression        : bool
                        OPTIONAL Speficies the problem type. Used for outputs etc (default False).
    model_filename    : String
                      : OPTIONAL Name of the persisted model files
    model_keep_epochs : bool
                        OPTIONAL Specifies intermediate (for every epoch) model persistency (default False).
    
    Methods
    -------
    fit()           :  trains a model
    evaluate()      :  validates the model
    output()        :  handles output and user feedback
             
    """
    def __init__(self, logger, model, loss_function, optimizer, 
                 scheduler = None, 
                 trainingstate = None, 
                 regression = False, 
                 model_filename = "model.pth.tar", 
                 model_keep_epochs = False):
        
        assert type(logger) == Logger2
        assert isinstance(model, nn.Module)
        assert isinstance(optimizer, torch.optim.Optimizer)
        assert isinstance(loss_function, nn.Module)
        assert isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler) if not scheduler is None else True
        assert type(trainingstate) == Trainingstate if not trainingstate is None else True
        
        self.name = "ummon.Trainer"

        # MEMBER VARIABLES
        # Training parameters
        self.model = model
        self.criterion = loss_function
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.trainingstate = Trainingstate()
        self.logger = Logger2()
        self.regression = regression
        self.epoch = 0
        
        # Persistence parameters
        self.model_filename = model_filename
        self.model_keep_epochs = model_keep_epochs
        
        # INITIALIZE LOGGER
        if logger:
            self.logger = logger
            
        # RESTORE STATE    
        if trainingstate:
            self.trainingstate = trainingstate
            self.logger.info("Loading Trainingstate.." )
            self.logger.info(self.trainingstate.get_summary())
            self.model.load_state_dict(trainingstate.state["model_state"])            
            self.optimizer.load_state_dict(trainingstate.state["optimizer_state"])
            self.epoch = self.trainingstate.state["training_loss[]"][-1][0]
        
    def fit(self, dataloader_training, validation_set, epochs, eval_interval, early_stopping, after_backward_hook=None, args=None):
        assert isinstance(dataloader_training, torch.utils.data.DataLoader)
        assert isinstance(validation_set, torch.utils.data.Dataset)
        
        if early_stopping:
            raise NotImplementedError("Early Stopping is not implemented yet!")
        
        # PRINT SOME INFORMATION ABOUT THE SCHEDULED TRAINING
        self.logger.print_problem_summary(self.model, self.criterion, self.optimizer, dataloader_training, validation_set, epochs, early_stopping)
        
        # COMPUTE BATCHES PER EPOCH
        batches = sum(1 for _ in iter(dataloader_training))
    
        for epoch in range(self.epoch, self.epoch + epochs):
            
            # TAKE TIME
            t = time.time()
	
            # ANNEAL LEARNING RATE
            if self.scheduler: self.scheduler.step()
           
            # Moving average
            n = 5
            avg_training_loss, avg_training_acc = None, None
            training_loss, training_acc  = np.zeros(n, dtype=np.float64), np.zeros(n, dtype=np.float64)
    
            # COMPUTE ONE EPOCH                
            for batch, data in enumerate(dataloader_training, 0):
            
                # Get the inputs
                inputs, targets = data
                
                # Execute Model
                output = self.model(Variable(inputs)).cpu()
                
                # Compute Loss
                targets = Variable(targets)
                loss = self.criterion(output, targets).cpu()
                
                # Zero the gradoemt    
                self.optimizer.zero_grad()
        
                # Backpropagation
                loss.backward()
                
                # Run hooks
                if after_backward_hook:
                    after_backward_hook()
                
                # Take gradient descent
                self.optimizer.step()
                
                # Running average training loss
                avg_training_loss = self._moving_average(batch, avg_training_loss, loss.data[0], training_loss)
                
                # Running average accuracy
                if not self.regression:
                    classes = Analyzer.classify(output)
                    acc = Analyzer.compute_accuracy(classes, targets)
                    avg_training_acc = self._moving_average(batch, avg_training_acc, acc, training_acc)
                else:
                    avg_training_acc = 0.
                    
                # Log status
                self.logger.log_one_batch(epoch + 1, batch + 1, batches, avg_training_loss, t)

            # Log epoch
            self.logger.log_epoch(epoch + 1, batch + 1, batches, avg_training_loss, dataloader_training.batch_size, t)

            # MODEL VALIDATION
            if (epoch +1) % eval_interval == 0:
                self.evaluate(epoch + 1, validation_set, avg_training_loss, avg_training_acc, dataloader_training.batch_size, args)
              
                
    def evaluate(self, epoch, validation_set, avg_training_loss, avg_training_acc, training_batch_size, args):
        # MODEL EVALUATION
        evaluation_dict = Analyzer.evaluate(self.model, self.criterion, validation_set, self.regression)
        
        # UPDATE TRAININGSTATE
        self.trainingstate.update_state(epoch, self.model, self.criterion, self.optimizer, 
                     training_loss = avg_training_loss, 
                     validation_loss = evaluation_dict["loss"], 
                     training_accuracy = avg_training_acc,
                     training_batchsize = training_batch_size,
                     validation_accuracy = evaluation_dict["accuracy"], 
                     validation_batchsize = len(validation_set),
                     args = args)

        self.logger.log_evaluation(self.trainingstate, evaluation_dict["samples_per_seconds"])

        # SAVE MODEL
        self.trainingstate.save_state(self.model_filename, self.model_keep_epochs)
        
        
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
    