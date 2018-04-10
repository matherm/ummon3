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
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
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
    trainingstate     : ummon.Trainingstate
                        OPTIONAL An optional trainingstate to initialize model and optimizer with an previous state
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
                 model_keep_epochs = False,
                 precision = np.float32,
                 use_cuda = False):
        
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
        self.regression = regression
        self.epoch = 0
        self.precision = precision
        self.use_cuda = use_cuda
        
        #  Persistency parameters
        self.model_filename = model_filename
        self.model_keep_epochs = model_keep_epochs
        
        # INITIALIZE LOGGER
        self.logger = logger
        
        # RESTORE STATE    
        if trainingstate:
            self.trainingstate = trainingstate
            self.logger.info("Loading training state.." )
            trs = self.trainingstate.get_summary()
            self.logger.info('Epochs: {}, best training loss ({}): {:.4f}, best validation loss ({}): {:.4f}'.format(
                trs['Epochs'], trs['Best Training Loss'][0], trs['Best Training Loss'][1], 
                trs['Best Validation Loss'][0], trs['Best Validation Loss'][1]))
            self.model.load_state_dict(trainingstate.state["model_state"])            
            self.optimizer.load_state_dict(trainingstate.state["optimizer_state"])
            self.epoch = self.trainingstate.state["training_loss[]"][-1][0]
            
            assert precision == self.trainingstate.state["precision"]
            self.precision = self.trainingstate.state["precision"]
        
        # Computational configuration
        if self.precision == np.float32:
            self.model = self.model.float()
        if self.precision == np.float64:
            self.model = self.model.double()
        if self.use_cuda:
            self.model = self.model.cuda()
        else:
            self.model = self.model.cpu()
    
    
    # check input data
    def _check_data(self, X, y=[]):
        '''
        Internal function for checking the validity and size compatibility of the provided 
        data.
        
        Arguments:
        
        * X: input data
        * y: output data (optional)
        
        '''
        # check inputs
        if type(X) != np.ndarray:
            self.logger.error('Input data is not a *NumPy* array')
        if X.ndim > 5 or X.ndim == 0:
            self.logger.error('Input dimension must be 1..5.', TypeError)
        if X.dtype != 'float32':
            X = X.astype('float32')
        
        # convert into standard shape
        if X.ndim == 1:
            X = X.reshape((1, len(X))).copy()
        elif X.ndim == 3:
            X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2])).copy()
        elif X.ndim == 4:
            X = X.reshape((X.shape[0], 1, X.shape[1], X.shape[2], X.shape[3])).copy()
        
        # check targets
        if len(y) > 0:
            if type(y) != np.ndarray:
                self.logger.error('Target data is not a *NumPy* array')
            if y.ndim > 2 or y.ndim == 0:
                self.logger.error('Targets must be given as vector or matrix.')
            if y.ndim == 1:
                y = y.reshape((1, len(y))).copy()
            if np.shape(y)[0] != np.shape(X)[0]:
                self.logger.error('Number of targets must match number of inputs.')
            if y.dtype != 'float32':
                y = y.astype('float32')
    
    
    def fit(self, dataloader_training, epochs=1, validation_set=None, eval_interval=500, 
        early_stopping=np.iinfo(np.int32).max, do_combined_retraining=False,
        after_backward_hook=None, after_eval_hook=None, args=None):
        
        # simple interface: training and test data given as numpy arrays
        if type(dataloader_training) == tuple:
            if len(dataloader_training) != 3:
                self.logger.error('Training data must be provided as a tuple (X,y,batch) or as PyTorch DataLoader.',
                    TypeError)
            
            # extract training data
            Xtr = dataloader_training[0]
            ytr = dataloader_training[1]
            self._check_data(Xtr, ytr)
            batch = int(dataloader_training[2])
            
            # construct pytorch dataloader from 2-tupel
            x = torch.from_numpy(Xtr)
            y = torch.from_numpy(ytr) 
            dataset = TensorDataset(x.float(), y.float())
            dataloader_training = DataLoader(dataset, batch_size=batch, shuffle=True, 
                sampler=None, batch_sampler=None)
            
        else:
            assert isinstance(dataloader_training, torch.utils.data.DataLoader)
        if validation_set is not None:
            assert isinstance(validation_set, torch.utils.data.Dataset)
        assert dataloader_training.dataset[0][0].numpy().dtype == self.precision
        
        # check parameters
        epochs = int(epochs)
        if epochs < 1:
            self.logger.error('Number of epochs must be > 0.', ValueError)
        eval_interval = int(eval_interval)
        early_stopping = np.int32(early_stopping)
        if early_stopping < np.iinfo(np.int32).max:
            raise NotImplementedError("Early Stopping is not implemented yet!")
        if early_stopping != np.iinfo(np.int32).max and validation_set is None:
            self.logger.error('Early stopping needs validation data.')
        do_combined_retraining = bool(do_combined_retraining)
        if do_combined_retraining:
            raise NotImplementedError("Combined retraining not implemented yet.")
        
        # PRINT SOME INFORMATION ABOUT THE SCHEDULED TRAINING
        self.logger.print_problem_summary(self.model, self.criterion, self.optimizer, 
            dataloader_training, validation_set, epochs, early_stopping)
        
        # COMPUTE BATCHES PER EPOCH
        batches = sum(1 for _ in iter(dataloader_training))
        
        for epoch in range(self.epoch, self.epoch + epochs):
            
            # TAKE TIME
            t = time.time()
            
            # ANNEAL LEARNING RATE
            if self.scheduler: 
                self.scheduler.step()
           
            # Moving average
            n = 5
            avg_training_loss, avg_training_acc = None, None
            training_loss, training_acc  = np.zeros(n, dtype=np.float64), np.zeros(n, 
                dtype=np.float64)
            
            # COMPUTE ONE EPOCH                
            for batch, data in enumerate(dataloader_training, 0):
            
                # Get the inputs
                inputs, targets = data
                
                # Handle cuda
                if self.use_cuda:
                    inputs = inputs.cuda()
                
                # Execute Model
                output = self.model(Variable(inputs)).cpu()
                
                # Compute Loss
                targets = Variable(targets)
                loss = self.criterion(output, targets).cpu()
                
                # Zero the gradient    
                self.optimizer.zero_grad()
        
                # Backpropagation
                loss.backward()
                
                # Run hooks
                if after_backward_hook is not None:
                    after_backward_hook(output.data, targets.data, loss.data)
                
                # Take gradient descent
                self.optimizer.step()
                
                # Running average training loss
                avg_training_loss = self._moving_average(batch, avg_training_loss, loss.data[0], training_loss)
                
                # Running average accuracy
                if not self.regression:
                    avg_training_acc = 0.
                    classes = Analyzer.classify(output.data)
                    acc = Analyzer.compute_accuracy(classes, targets.data)
                    avg_training_acc = self._moving_average(batch, avg_training_acc, acc, training_acc)
                else:
                    avg_training_acc = 0.
                    
                # Log status
                self.logger.log_one_batch(epoch + 1, batch + 1, batches, avg_training_loss, t)
            
            # Log epoch
            self.logger.log_epoch(epoch + 1, batch + 1, batches, avg_training_loss, dataloader_training.batch_size, t)
            
            # MODEL VALIDATION
            if validation_set is not None and (epoch +1) % eval_interval == 0:
                self.evaluate(epoch + 1, validation_set, avg_training_loss, avg_training_acc, 
                              dataloader_training.batch_size, dataloader_training, after_eval_hook, args)
                
        return self.trainingstate
              
                
    def evaluate(self, epoch, validation_set, avg_training_loss, avg_training_acc, 
        training_batch_size, dataloader_training, after_eval_hook, args):
        # INIT ARGS
        args = {} if args is None else args
        
        # MODEL EVALUATION
        evaluation_dict = Analyzer.evaluate(self.model, self.criterion, validation_set, self.regression, after_eval_hook)
        
        # UPDATE TRAININGSTATE
        self.trainingstate.update_state(epoch, self.model, self.criterion, self.optimizer, 
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
        
        self.logger.log_evaluation(self.trainingstate)
        
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
    