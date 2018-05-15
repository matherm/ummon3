#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################

"""

ummon3 Examples

MNIST 1-validation

Run command:
    
    python mnist1-validation.py

"""
              

#
# IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from ummon.trainingstate import Trainingstate
from ummon.logger import Logger
from ummon.supervised import ClassificationAnalyzer

#
# SET inital seed for reproducibility 
np.random.seed(17)
torch.manual_seed(17)


#
# DEFINE a neural network
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 84)
        self.fc3 = nn.Linear(84, 10)
        
        # Initialization
        def weights_init_normal(m):
            if type(m) == nn.Linear:
                nn.init.normal(m.weight, mean=0, std=0.1)
            if type(m) == nn.Conv2d:
                nn.init.normal(m.weight, mean=0, std=0.1)
        self.apply(weights_init_normal)
    
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class DefaultValues(dict):
    def __init__(self):
        dict.__init__(self, {
                        "use_cuda" : False,
                        "model" : "MNIST1.pth.tar"
                        })
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__  


def example(argv = DefaultValues()):
    
        # PREPARE TEST DATA
        Xtr = np.random.randn(10*28*28).reshape(10,28,28).astype(np.float32)
        ytr = np.arange(10, dtype=np.int64)
        
        # CHOOSE MODEL
        model = Net()  
        
        # CHOOSE LOSS-Function
        criterion = nn.CrossEntropyLoss()
        
        # LOAD TRAINING STATE
        if argv.model is not "":
            ts = Trainingstate(argv.model)
            ts.load_weights_(model, None)
        
        with Logger(logdir='.', log_batch_interval=500) as lg:
            lg.info(ClassificationAnalyzer.evaluate(model, criterion, (Xtr, ytr)))
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ummon3 - example - MNIST 1 (Validation Mode)')
    #
    # TRAINING PARAMS
    parser.add_argument('--use_cuda', action='store_true', dest='use_cuda',
                        help="Shall cuda be used (default: False)")
    parser.add_argument('--model', default="MNIST1", metavar="",
                        help="Path to trained model")
    
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]     
    
    example(argv)
  
            
    
