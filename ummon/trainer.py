##########################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
##########################################################################

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
    def __init__(self, logger):
        self.name = "ummon.Trainer"
        
        if logger:
            self.logger = logger
        else:
            self.logger = Logger()
        
    def fit(self, model, dataloader_training, dataloader_test, optimizer, scheduler, epochs, early_stopping):
        print("fit()")
                
    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == "__main__":
    print("This is", Trainer().name)
    