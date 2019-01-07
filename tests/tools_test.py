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
import ummon.tools.stateviewer
from ummon.schedulers import *
from ummon.trainingstate import *
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

class TestTools(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestTools, self).__init__(*args, **kwargs)
        
        # BACKUP files
        backup_dir = "_backup"
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension) or file.endswith(".log"):
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                os.rename(os.path.join(dir,file), os.path.join(backup_dir,file))

    @staticmethod
    def _sample_state(model):
                #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(10, 256)
                
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.softmax(x, dim=1)
                    return x
        
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(Net().parameters(), lr=0.01)
        
        ts = Trainingstate()
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = 0, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     validation_dataset = dataset_valid,
                     args = { "args" : 1 , "argv" : 2})
        ts.save_state(model)
    
    def test_stateviewer(self):
        TestTools._sample_state(model="test.pth.tar")
        ummon.tools.stateviewer.view("test")
        ummon.tools.stateviewer.view("test.pth.tar")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('test', default="", metavar="",
                        help="Execute a specific test")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]
    if argv.test is not "":
        eval(str("TestTools()." + argv.test + '()'))
    else:
        unittest.main()
