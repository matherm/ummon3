#---------------------------------------------------------------
import os, sys 
sys.path.insert(0, os.getcwd()) # enables $ python examples/[EXAMPLE].py
#---------------------------------------------------------------

"""

ummon3 Examples

SINE

Run command:
    
    python sine.py

"""

#
# IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ummon import *
#
# SET inital seed for reproducibility 
np.random.seed(17)
torch.manual_seed(17)

#
# DEFINE a neural network
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 5)
        self.fc2 = nn.Linear(5, 1)
        
        # Initialization
        def weights_init_normal(m):
            if type(m) == nn.Linear:
                nn.init.normal_(m.weight, mean=0, std=0.1)
        self.apply(weights_init_normal)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DefaultValues(dict):
    def __init__(self):
        dict.__init__(self, {
                        "epochs" : 1,
                        "lrate": 0.01,
                        "use_cuda" : False,
                        "batch_size" : 2,
                        "view" : ""
                        })
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__  
    
def example(argv = DefaultValues()):
   
    if argv.view is not "":
        ts = Trainingstate(argv.view)
        print(ts.get_summary())
    else:
        # PREPARE TEST DATA
        Xtr = np.arange(1000).reshape(1000,1).astype(np.float32)
        ytr = np.sin(Xtr) + np.random.normal(0,1, size=Xtr.shape).astype(np.float32)

        Xts = np.arange(100).reshape(100,1).astype(np.float32)
        yts = np.sin(Xts)
        
        # CHOOSE MODEL
        model = Net()  
        
        # CHOOSE LOSS-Function
        criterion = nn.MSELoss(reduction="sum")
        
        # INSTANTIATE OPTIMIZER
        optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate/argv.batch_size)
        
        # LOAD TRAINING STATE
        try:
            ts = Trainingstate("SINE.pth.tar")
        except FileNotFoundError:
            ts = Trainingstate()
        
        with Logger(loglevel=20, logdir='.', log_batch_interval=100) as lg:
            
            # CREATE A TRAINER
            my_trainer = SupervisedTrainer(lg, 
                                model, 
                                criterion, 
                                optimizer, 
                                trainingstate=ts,
                                model_filename="SINE", 
                                precision=np.float32,
                                use_cuda=argv.use_cuda)
            
            
            # START TRAINING
            my_trainer.fit(dataloader_training=(Xtr, ytr, argv.batch_size),
                                        epochs=argv.epochs,
                                        validation_set=(Xts, yts))
            
        return ts

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ummon3 - example - Sine')
    #
    #
    # TRAINING PARAMS
    parser.add_argument('--epochs', type=int, default=5, metavar='',
                        help='Amount of epochs for training (default: 2)')
    parser.add_argument('--batch_size', type=int, default=2, metavar='',
                        help='Batch size for SGD (default: 40)')
    parser.add_argument('--lrate', type=float, default=0.01, metavar="",
                        help="Learning rate (default: 0.01")
    parser.add_argument('--use_cuda', action='store_true', dest='use_cuda',
                        help="Shall cuda be used (default: False)")
    parser.add_argument('--view', default="", metavar="",
                        help="Print summary about a trained model")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]                    

    
    example(argv)

            
    
