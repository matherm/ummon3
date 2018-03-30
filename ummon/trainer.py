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
from ummon.logger import Logger
from ummon.trainingstate import Trainingstate
from ummon.analyzer import Analyzer

class Trainer:
    """
    This class provides a generic trainer for training PyTorch-models.
    
    Constructor
    -----------
    logger      :   ummon.Logger
                    The logger to use (if NULL default logger will be used)
    model       :   torch.nn.module
                    The model used for training
    optimizer   :   torch.optim.optimizer
                    The optimizer for the training
    
    OPTIONAL
    
    trainingstate : ummon.Trainingstate
                    An optional trainingstate to initialize model and optimizer with an previous state
             
    Methods
    -------
    fit()           :  trains a model
    evaluate()      :  validates the model
    output()        :  handles output and user feedback
             
    """
    def __init__(self, logger, model, loss_function, optimizer, 
                 scheduler = None, 
                 trainingstate = None, 
                 is_classifier = True, 
                 model_filename = "model", 
                 model_keep_epochs = False):
        
        assert type(logger) == Logger
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
        self.logger = Logger()
        self.is_classifier = is_classifier
        
        # Persistence parameters
        self.model_filename = model_filename
        self.model_keep_epochs = model_keep_epochs
        
        # INITIALIZE LOGGER
        if logger:
            self.logger = logger
            
        # RESTORE STATE    
        if trainingstate:
            self.model.load_state_dict(trainingstate.state["model_state"])            
            self.optimizer.load_state_dict(trainingstate.state["optimizer_state"])
            self.trainingstate = trainingstate
        
    def fit(self, dataloader_training, validation_set, epochs, eval_interval, early_stopping, after_backward_hook=None, args=None):
        assert isinstance(dataloader_training, torch.utils.data.DataLoader)
        assert isinstance(validation_set, torch.utils.data.Dataset)
        
        if early_stopping:
            raise NotImplementedError("Early Stopping is not implemented yet!")
        
        # PRINT SOME INFORMATION ABOUT THE SCHEDULED TRAINING
        self.logger.print_problem_summary(self.model, self.criterion, self.optimizer, dataloader_training, validation_set)
        
        # COMPUTE BATCHES PER EPOCH
        batches = sum(1 for _ in iter(dataloader_training)) - 1
    
        for i_epoch in range(epochs):
            
            # TAKE TIME
            t = time.time()
	
            # ANNEAL LEARNING RATE
            if self.scheduler: self.scheduler.step()
    
            avg_training_loss = None
            avg_training_acc = None
    
            # COMPUTE ONE EPOCH                
            for i, data in enumerate(dataloader_training, 0):
            
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
                if avg_training_loss is not None:
                    avg_training_loss = avg_training_loss + loss / (i + 1)
                else:
                    avg_training_loss = loss
                
                # Running average accuracy
                if self.is_classifier:
                    classes = Analyzer.classify(output)
                    acc = Analyzer.compute_accuracy(classes, targets)
                    if avg_training_acc is not None:
                        avg_training_acc = avg_training_acc + acc / (i + 1)
                    else:
                        avg_training_acc = acc
                
                
                # Log status
                self.logger.log_one_batch(i_epoch, i, batches, avg_training_loss, t)

            # MODEL VALIDATION
            if i_epoch % eval_interval == 0:
                self.evaluate(i_epoch, validation_set, avg_training_loss, avg_training_acc, dataloader_training.batch_size, args)
              
                
    def evaluate(self, epoch, validation_set, avg_training_loss, avg_training_acc, training_batchsize, args):
        # MODEL EVALUATION
        evaluation_dict = Analyzer.evaluate(self.model, self.criterion, validation_set)
        
        # UPDATE TRAININGSTATE
        self.trainingstate.update_state(epoch, self.model, self.criterion, self.optimizer, 
                     training_loss = avg_training_loss, 
                     validation_loss = evaluation_dict["loss"], 
                     test_loss = None, 
                     training_accuracy = avg_training_acc,
                     training_batchsize = training_batchsize,
                     validation_accuracy = evaluation_dict["accuracy"], 
                     validation_batchsize = len(validation_set),
                     test_accuracy = None, 
                     test_batchsize = 0,
                     args = args)
        
        # SAVE MODEL
        self.trainingstate.save_state(self.model_filename, self.model_keep_epochs)
        
        
if __name__ == "__main__":
    print("This is", Trainer().name)
    