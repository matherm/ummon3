##########################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
##########################################################################

import numpy as np
import torch
import ummon.logger
import ummon.analyzer

class Trainer:
    def __init__(self):
        self.name = "ummon.Trainer"
        
    def fit(self):
        print("fit()")
              
    def save(self, filename):
        pass


if __name__ == "__main__":
    print("This is", Trainer().name)
    