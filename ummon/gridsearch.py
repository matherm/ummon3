import os
import numpy as np
import torch.nn as nn
import torch.utils.data
import ummon.utils as uu
from .logger import Logger

__all__ = ["GridSearch"]

class GridSearch:
    """
    This class provides a grid search implementation.
    Basic usage is that a trainer class and lists of parameters are given.
    The class then searches the space and responds with a dictionary.
    Additionally trained models are stored.
    
    Methods
    -------
    grid_search()         : Computes a grid search
    """

    @staticmethod
    def grid_search(trainer, dataloader_training, dataset_valid,
                    epochs = [100],
                    logger=Logger()):
        """
        Arguments
        ---------
        *trainer (list<instances: ummon.MetaTrainer>) : The trainer class to use.
        *dataloader_training (torch.utils.data.DataLoader OR tuple (X,y,batch)) : Training data.
        *dataset_valid (torch.utils.data.Dataset) : Validation data.
        *epochs (int) : epochs to train.
        *filepath (str) : Path where to store trained models.
        *logger (ummon.Logger) : The logger to use.

        Return
        ------
        *dictionary
        """
        #Final dictionary
        result = {}
        
        logger.info("Starting Grid-Search with ", len(trainer), "trainer instances.")
        for i, _trainer in enumerate(trainer):

            logger.info("Trainer: ", i)

            for _epochs in epochs:
                
                # Fit the model
                _trainer.fit(dataloader_training, epochs=_epochs, validation_set=None, eval_batch_size=-1)

                # Evaluate model
                result["Training 1"] = _trainer.trainingstate.get_loss()

        return result




       
