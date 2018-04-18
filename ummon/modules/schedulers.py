import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer


__all__ = [ 'StepLR_earlystop' ]


class StepLR_earlystop(object):
    '''
    Sets the learning rate of each parameter group to the initial lr
    decayed by gamma every step_size epochs. When last_epoch=-1, sets
    initial lr as lr.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        step_size (int): Period of learning rate decay.
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        gamma (float): Multiplicative factor of learning rate decay. Default: 0.1.
        patience (int): Number of epochs with no improvement after
            which learning rate will be reduced. Default: 10.
    
    The step function of this scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced (early stopping). The step returns a boolean
    value which is True when the scheduler switches to the next learning rate. This can be
    used to reload the currently best model before resuming training.
    
    Example:
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60 or if previous step was aborted by early stopping
        >>> # lr = 0.0005   if 60 <= epoch < 90 or if previous step was aborted by early stopping
        >>> # ...
        >>> scheduler = StepLR_earlystop(optimizer, step_size=30, mode='min', gamma=0.1, patience=5)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     val_loss = validate(...)
        >>>     # Note that step should be called after validate()
        >>>     switched = scheduler.step(val_loss)
        >>>     if switched:
        >>>         reload current best state
    '''
    def __init__(self, optimizer, step_size, mode='min', gamma=0.1, patience=10):
        
        # check arguments
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
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
        self.last_epoch = 1 # we assume that step() is called the first time after training in the first epoch
        if self.mode == 'min':
            self.best = np.iinfo(np.float32).max
        else:
            self.best = np.iinfo(np.float32).min
        self.num_bad_epochs = 0
        self.num_epochs_in_step = 0
    
    
    # reduce learning rate and reset internal counters
    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group['lr'])
            new_lr = old_lr * self.gamma
            param_group['lr'] = new_lr
        self.num_bad_epochs = 0
        self.num_epochs_in_step = 0
    
    
    # scheduler step, returns True when next learning rate is set
    def step(self, metrics):
        
        # increase epoch counters
        self.last_epoch += 1
        self.num_epochs_in_step += 1
        
        # new best value
        if self.mode == 'min' and metrics < self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        elif self.mode == 'max' and metrics > self.best:
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1
        
        # early stopping
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            return True
        
        # end of current learning rate reached => go to next learning rate
        if self.num_epochs_in_step == self.step_size:
            self._reduce_lr(epoch)
            return True
        
        return False

