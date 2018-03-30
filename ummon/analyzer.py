#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################


class Analyzer:
    """
    This class provides a generic analyzer for PyTorch-models. For a given PyTorch-model it 
    computes statistical information about the model, e.g. accuracy, loss, ROC, etc.
    
    
    Constructor
    -----------
    model           :   torch.nn.module
                        The torch module to use
    training_state  :   ummon.trainingstate (dictionary)
                        The training state
             
    Methods
    -------
    predict()           :  Predicts any given data
    accuracy()          :  Computes accuracy            
             
    """
    def __init__(self, model, training_state):
        self.name = "ummon.Analyzer"
   
    def load(self, model, training_state_filename):
        pass

    def predict(model, dataset):
        pass

if __name__ == "__main__":
    print("This is", Analyzer().name)
    
    
    
    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 16:17:12 2018

@author: matthias
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REINFORCE Implementation

[1] Williams, R. J. (1992). 
    Simple statistical gradient-following methods for connectionist reinforcement learning. 
    Machine Learning, 8, 229–256. https://doi.org/10.1007/BF00992696


Created on Mon Jan 15 13:01:27 2018

@author: matthias
@copyright: IOS
"""

import argparse
import torch
import image_transformations

from model import RVA
from torchvision.datasets import MNIST
from glimpse_sensor import GlimpseSensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys

parser = argparse.ArgumentParser(description='PyTorch Recurrent Model for Visual Attention')
parser.add_argument('--state', default="model_best.pth.tar", metavar="",
                    help="The state file (default: model_best.pth.tar)")
parser.add_argument('--plot', action='store_true', dest='plot',
                    help="Shall python plot intermediate tests with matplotlib (default: False)")
parser.add_argument('--use_cuda', action='store_true', dest='use_cuda',
                    help="Shall cuda be used (default: False)")
parser.add_argument('--view', action='store_true', dest='view', 
                    help="Just view the saved state (default: False)")
argv = parser.parse_args()

sys.argv = [sys.argv[0]]
import train as train


if __name__ == '__main__':
    
# Load the state
if argv.use_cuda:
    state = torch.load(argv.state)
else:
    #state = torch.load("model_best.pth.tar", map_location=lambda storage, loc: storage)
    #plt.plot(state["num_epoch[]"], state["avg_scale[]"])
    state = torch.load(argv.state, map_location=lambda storage, loc: storage)

# COMBINE ARGUMENTS
saved_args = state["args"]
saved_args.plot = argv.plot
saved_args.use_cuda = argv.use_cuda         

# MNIST
if saved_args.mode == 0:
    mnist_data_test = MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)
   
# TRANSLATED MNIST
if saved_args.mode == 1:    
    transform = transforms.Compose([transforms.ToTensor(), image_transformations.to_translated ])
    mnist_data_test = MNIST("./", train=True, transform=transform, target_transform=None, download=True)

testloader = DataLoader(mnist_data_test, batch_size=len(mnist_data_test), shuffle=True, sampler=None, batch_sampler=None)  
   
# INSTANTIATE ENVIRONMENT
glimpse_sensor = GlimpseSensor(glimpse_size=(saved_args.glimpse_size, saved_args.glimpse_size), levels=(saved_args.glimpse_level, 2),  enable_history=True, debug=False, optimized=True)                

# CHOOSE MODEL
model = RVA(args=saved_args, glimpse_sensor=glimpse_sensor)   
model.load_state_dict(state['model'])

train.print_summary(None, testloader, saved_args, model)
   
print("\n[State]")
print("Epochs:\t\t", state["num_epoch[]"][-1])
print("Best Error:\t {:2.2f} %".format(state["best_error"]))
print("Best Total Loss: {:2.4f}".format(state["best_total_loss"]))

if not argv.view:
    print("\nStarting Inference...\n")
    train.evaluate(testloader, model, learning_state=None, args=saved_args)
    
