import numpy as np
import torch
import torch.nn as nn
import copy
from torch.optim import Optimizer

from ummon.logger import Logger


__all__ = [ 'StepLR_earlystop', 'StepsFinished' ]

class StepsFinished(Exception):
    pass


class StepLR_earlystop(object):
    '''
    Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. 
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        trs (Trainingstate): state of the Trainer class that uses the scheduler
        model (Module): neural network or layer
        step_size (int): Number of epochs until learning rate is decreased.
        nsteps (int): number of learning rates
        logger (Logger): Logger
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
        relative_improvement (float): Factor of relative improvement of the evaluated 
            metrics for early stopping, i.e. new metrics has to be < best - best * factor
            (if mode `min`).
    
    The step function of this scheduler reads a peformance metric from the state dict
    of the trainer. If no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced (early stopping). At the same time, the
    weights of the currently best model are reloaded into the provided network or layer
    so that training is continued from this point in the next step. The scheduler can
    be reset, e.g. when you do a combined training with training and validation data
    after you finish the stepwise schedule. After the final number of 'nsteps' lerning 
    rates has been reached, the scheduler raises an exception of type 'StepsFinished'
    which should be caught by the calling routine.
    
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60 or if previous step was aborted by early stopping
        >>> # lr = 0.0005   if 60 <= epoch < 90 or if previous step was aborted by early stopping
        >>> # ...
        >>> scheduler = StepLR_earlystop(optimizer, training_state, model, step_size=30, 
        >>>     nsteps=2, logger=lg, mode='min', gamma=0.1, patience=5)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     # Note that step should be called after validate()
        >>>     try:
        >>>         scheduler.step()
        >>>     except StepsFinished:
        >>>          break
    '''
    def __init__(self, optimizer, trs, model, step_size, nsteps, logger, mode='min', 
        gamma=0.1, patience=10, relative_improvement=0):
        
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
        self.nsteps = int(nsteps)
        if self.nsteps < 1:
            raise ValueError('Number of steps must be > 0.')
        if not isinstance(logger, Logger):
            raise TypeError('{} is not a Logger'.format(
                type(logger).__name__))
        self.logger = logger
        self.mode = str(mode)
        if self.mode not in ['max', 'min']:
            raise ValueError('Optimization mode must be "min" or "max".')
        self.gamma = float(gamma)
        if self.gamma <= 0.0:
            raise ValueError('Gamma must be > 0.')        
        self.patience = int(patience)
        if self.patience < 1:
            raise ValueError('Early stopping patience must be > 0.')
        self.relative_improvement = float(relative_improvement)
        if self.relative_improvement < .0:
            raise ValueError('Relative improvement must be >= 0.')
        
        # init scheduler state
        self.last_epoch = 0
        self.num_epochs_since_last_eval = 0
        self.lr = []
        for param_group in self.optimizer.param_groups:
            self.lr.append(float(param_group['lr']))
        if self.nsteps > 1:
            self.logger.info('Scheduler: {} learning rates decreased by factor {} after {} epochs, early stopping after {}, {} mode.'.format(
                self.nsteps, self.gamma, self.step_size, self.patience, self.mode))
        else:
            self.logger.info('Scheduler: lr={}, {} epochs, early stopping after {}, {} mode.'.format(
                self.lr[0], self.step_size, self.patience, self.mode))
            
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
        self.cur_step = 0
    
    
    def _reduce_lr(self):
        '''
        Reduces learning rate of optimizer by factor gamma.
        '''
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group['lr'] = self.lr[i] * (self.gamma**self.cur_step)
    
    
    def _next_lrstep(self):
        '''
        Next learning rate step: decrease learning rate, reset parameters, reload best
        model and stop training if last step reached.
        '''
        self.logger.info('Current best performance for learning rate {:1.5f}: {:4.5f}.'.format(
            self.trs.current_lrate(), self.best))
        self.num_bad_epochs = 0
        self.num_epochs_in_step = 0
        self.cur_step += 1
        self.trs.maybe_load_best_available_model_(self.model, self.optimizer)
        if self.cur_step >= self.nsteps:
            self.logger.info('Maximum number of learning rate steps reached.')
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.lr[i] * (self.gamma**(self.cur_step - 1))
            raise StepsFinished 
        self._reduce_lr()
    
    
    def step(self):
        '''
        Scheduler step.
        '''
        # increase epoch counters
        self.last_epoch += 1
        self.num_epochs_in_step += 1
        
        # Check if there was an evaluation run since the last time we checked
        last_eval_epoch = self.trs.current_epoch()
        if last_eval_epoch < self.last_epoch:
            self.num_epochs_since_last_eval += 1
        else:
            # new best value
            metrics = self.trs.current_validation_loss() # get current validation loss
            if metrics is None:
                metrics = self.trs.current_training_loss() # get current training loss
            if self.mode == 'min' and metrics < (self.best - self.best * self.relative_improvement):
                self.best = metrics
                self.num_bad_epochs = 0
            elif self.mode == 'max' and metrics > (self.best + self.best * self.relative_improvement):
                self.best = metrics
                self.num_bad_epochs = 0
            else:
                self.num_bad_epochs += 1 + self.num_epochs_since_last_eval
            self.num_epochs_since_last_eval = 0
            
            # early stopping
            if self.num_bad_epochs >= self.patience:
                self.logger.info("No improvement since {} epochs. Stopping early and reloading current best model.".format(
                            self.patience))
                self._next_lrstep()
        
        # end of current learning rate reached => go to next learning rate
        if self.num_epochs_in_step == self.step_size:
            self.logger.info('Learning rate step finished. Reloading current best model.')
            self._next_lrstep()
    
    
    def load_state_dict(self, state_dict):
        '''
        Returns the state of the scheduler as a :class:`dict`.
        '''
        # deepcopy, to be consistent with module API
        state_dict = copy.deepcopy(state_dict)
        
        self.step_size = state_dict['step_size']
        self.nsteps = state_dict['nsteps']
        self.mode = state_dict['mode']
        self.gamma = state_dict['gamma']
        self.patience = state_dict['patience']
        self.last_epoch = state_dict['last_epoch']
        self.num_epochs_since_last_eval = state_dict['num_epochs_since_last_eval']
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']
        self.num_epochs_in_step = state_dict['num_epochs_in_step']
        self.cur_step = state_dict['cur_step']
        self.lr = state_dict['lr']
    
    
    def state_dict(self):
        '''
        Loads the scheduler state.
        
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        '''
        return {
            'step_size': self.step_size,
            'nsteps': self.nsteps,
            'mode': self.mode,
            'gamma': self.gamma,
            'patience': self.patience,
            'last_epoch': self.last_epoch,
            'num_epochs_since_last_eval': self.num_epochs_since_last_eval,
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
            'num_epochs_in_step': self.num_epochs_in_step,
            'cur_step': self.cur_step,
            'lr': self.lr
        }

from ..trainingstate import Trainingstate
