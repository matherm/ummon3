#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import unittest
import logging
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import ummon.utils as uu
from ummon.schedulers import *
from ummon.trainingstate import *
from ummon.data import *
from ummon.trainer import *
from ummon.unsupervised import *
from ummon.supervised import *
from ummon.logger import *
from ummon.trainingstate import *
from ummon.analyzer import *
from ummon.visualizer import *
from ummon.modules.container import *
from ummon.modules.linear import *

# set fixed seed for reproducible results
torch.manual_seed(4)
np.random.seed(4)

class TestExamples(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestExamples, self).__init__(*args, **kwargs)
        
        # BACKUP files
        backup_dir = "_backup"
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension) or file.endswith(".log"):
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                os.rename(os.path.join(dir,file), os.path.join(backup_dir,file))
    
    def test_mnist1(self):
        import examples.mnist_pytorch_examples
        
        ts = examples.mnist_pytorch_examples.example()
        assert ts["best_validation_accuracy"][1] > 0.11
        
        import examples.validation
        examples.validation.example()
        
        # Clean up
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension) or file.endswith(".log"):
                os.remove(os.path.join(dir,file))
         
    def test_sine(self):
        import examples.sine

        ts = examples.sine.example()
        assert ts["best_validation_loss"][1] < 1.2
        
         # Clean up
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension) or file.endswith(".log"):
                os.remove(os.path.join(dir,file))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default="", metavar="",
                        help="Execute a specific test")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]
    if argv.test is not "":
        eval(str("TestExamples()." + argv.test + '()'))
    else:
        unittest.main()
