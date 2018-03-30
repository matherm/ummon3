#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################

import numpy as np
import shutil
import torch
from ummon.logger import Logger
import ummon.analyzer

class Trainer:
    """
    This class provides a generic trainer for training PyTorch-models.
    
    Constructor
    -----------
    logger : ummon.Logger
             The logger to use (if NULL default logger will be used)       
             
    Methods
    -------
    fit()               :  trains a model
    save_checkpoint()   :  saves a checkpoint of the model to file            
             
    """
    def __init__(self, logger, model):
        self.name = "ummon.Trainer"
        
        # MEMBER VARIABLES
        if logger:
            self.logger = logger
        else:
            self.logger = Logger()
        self.model = model
        
    def fit(self, dataloader_training, dataloader_test, optimizer, scheduler, epochs, early_stopping):
        print("fit()")
                
    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
    
    def load_checkpoint(self, checkpoint):
        self.model.load_state_dict(state['model'])

if __name__ == "__main__":
    print("This is", Trainer().name)
    