##########################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
##########################################################################

from ummon.trainer import Trainer

if __name__ == "__main__":
    
    # Prepare the trainer
    trainer = Trainer()
    
    # Fit the model
    #trainer.fit(model, X, y, loss, optimizer, logger)
    