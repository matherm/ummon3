import numpy as np
import torch
import torch.nn as nn
import copy
from torch.optim import Optimizer


__all__ = [ 'StepLR_earlystop' ]


class StepLR_earlystop(object):
    '''
    Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. 
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        trs (Trainingstate): state of the Trainer class that uses the scheduler
        model (Module): neural network or layer
        step_size (int): Period of learning rate decay.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
    
    The step function of this scheduler reads a peformance metric from the state dict
    of the trainer.If no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced (early stopping). At the same time, the
    weights of the currently best model are reloaded into the provided network or layer
    so that training is continued from this point in the next step. The scheduler can
    be reset, e.g. when you do a combined training with training and validation data
    after you finish the stepwise schedule.
    
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60 or if previous step was aborted by early stopping
        >>> # lr = 0.0005   if 60 <= epoch < 90 or if previous step was aborted by early stopping
        >>> # ...
        >>> scheduler = StepLR_earlystop(optimizer, training_state, model, step_size=30, 
        >>>     mode='min', gamma=0.1, patience=5)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     # Note that step should be called after validate()
        >>>     scheduler.step()
    '''
    def __init__(self, optimizer, trs, model, step_size, mode='min', gamma=0.1, 
        patience=10):
        
        # check arguments
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if not isinstance(trs, Trainingstate):
            raise TypeError('{} is not a training state'.format(
                type(trs).__name__))
        self.trs = trs
        if not isinstance(model, nn.Module):
            raise TypeError('{} is not a nn.Module'.format(
                type(model).__name__))
        self.model = model
        self.step_size = int(step_size)
        if self.step_size < 1:
            raise ValueError('Step size must be > 0.')
        self.mode = str(mode)
        if self.mode not in ['max', 'min']:
            raise ValueError('Optimization mode must be "min" or "max".')
        self.gamma = float(gamma)
        if self.gamma <= 0.0:
            raise ValueError('Gamma must be > 0.')        
        self.patience = int(patience)
        if self.patience < 1:
            raise ValueError('Early stopping patience must be > 0.')   
        
        # init scheduler state
        self.last_epoch = 0 
        self.reset()
    
    
    def reset(self):    
        '''
        Reset scheduler to initial state.
        '''
        if self.mode == 'min':
            self.best = np.finfo(np.float32).max
        else:
            self.best = np.finfo(np.float32).min
        self.num_bad_epochs = 0
        self.num_epochs_in_step = 0
    
    
    def _reduce_lr(self):
        '''
        Reduces learning rate of optimizer by factor gamma.
        '''
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.gamma
            param_group['lr'] = new_lr
    
    
    def step(self):
        '''
        Scheduler step.
        '''
        # increase epoch counters
        self.last_epoch += 1
        self.num_epochs_in_step += 1
        
        # new best value
        metrics = self.trs['validation_loss[]'][-1][1] # get current validation loss
        if self.mode == 'min' and metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        elif self.mode == 'max' and metrics > self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # early stopping
        if self.num_bad_epochs >= self.patience:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.num_epochs_in_step = 0
            self.trs.load_weights_best_validation(self.model)
        
        # end of current learning rate reached => go to next learning rate
        if self.num_epochs_in_step == self.step_size:
            self._reduce_lr()
            self.num_bad_epochs = 0
            self.num_epochs_in_step = 0
            self.trs.load_weights_best_validation(self.model)
            
    
    def load_state_dict(self, state_dict):
        '''
        Returns the state of the scheduler as a :class:`dict`.
        '''
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        
        self.optimizer = state_dict['optimizer']
        self.step_size = state_dict['step_size']
        self.mode = state_dict['mode']
        self.gamma = state_dict['gamma']
        self.patience = state_dict['patience']
        self.last_epoch = state_dict['last_epoch']
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.num_epochs_in_step = state_dict['num_epochs_in_step']
    
    
    def state_dict(self):
        '''
        Loads the scheduler state.
        
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        '''
        return {
            'optimizer': self.optimizer,
            'step_size': self.step_size,
            'mode': self.mode,
            'gamma': self.gamma,
            'patience': self.patience,
            'last_epoch': self.last_epoch,
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'num_epochs_in_step': self.num_epochs_in_step
        }

from .trainingstate import Trainingstate
