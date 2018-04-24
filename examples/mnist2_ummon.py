#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################

"""

ummon3 Examples

MNIST 1

Run command:
    
    python MNIST1.py --epochs 1 --batch_size 40

"""
#
# IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ummon import Trainingstate
from ummon import Logger
from ummon import ClassificationTrainer
import load_mnist

#
# SET inital seed for reproducibility 
np.random.seed(17)
torch.manual_seed(17)

#
# DEFINE a neural network
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 30)
        self.fc2 = nn.Linear(30, 10)
        
        # Initialization
        def weights_init_normal(m):
            if type(m) == nn.Linear:
                nn.init.normal(m.weight, mean=0, std=0.1)
        self.apply(weights_init_normal)
    
    def forward(self, x):
        x = F.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
    

class DefaultValues(dict):
    def __init__(self):
        dict.__init__(self, {
                        "epochs" : 1,
                        "lrate": 0.01,
                        "use_cuda" : False,
                        "batch_size" : 40,
                        "view" : "",
                        "eval_interval" : 1
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
        X0,y0,Xv,yv,X1,y1 = load_mnist.read([0,1,2,3,4,5,6,7,8,9], path="")
        x0 = (1.0/255.0)*X0.astype('float32')
        x1 = (1.0/255.0)*X1.astype('float32')
        x2 = (1.0/255.0)*Xv.astype('float32')
        y0 = y0.astype('float32')
        y1 = y1.astype('float32')
        y2 = yv.astype('float32')   

        # CHOOSE MODEL
        model = Net()  
        
        # CHOOSE LOSS-Function
        loss = nn.BCEWithLogitsLoss(size_average = False)          

        # INSTANTIATE OPTIMIZER
        optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate / argv.batch_size)
        
        # LOAD TRAINING STATE
        try:
            ts = Trainingstate("MNIST2.pth.tar")
        except FileNotFoundError:
            ts = Trainingstate()
        
        with Logger(logdir='.', log_batch_interval=5000) as lg:
            
            # CREATE A TRAINER
            my_trainer = ClassificationTrainer(lg, 
                                model, 
                                loss, 
                                optimizer, 
                                model_filename="MNIST2", 
                                precision=np.float32,
                                combined_training_epochs = 0,
                                use_cuda=argv.use_cuda,
                                profile=False)
            
            # START TRAINING
            my_trainer.fit((x0,y0,argv.batch_size), 
                           epochs=argv.epochs, 
                           validation_set=(x2,y2), 
                           eval_interval=argv.eval_interval)
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ummon3 - example - MNIST 1')
    #
    # TRAINING PARAMS
    parser.add_argument('--epochs', type=int, default=10, metavar='',
                        help='Amount of epochs for training (default: 2)')
    parser.add_argument('--batch_size', type=int, default=16, metavar='',
                        help='Batch size for SGD (default: 16)')
    parser.add_argument('--eval_interval', type=int, default=5, metavar='',
                        help='Evaluate model in given interval (epochs) (default: 1)')
    parser.add_argument('--lrate', type=float, default=0.1, metavar="",
                        help="Learning rate (default: 0.01")
    parser.add_argument('--use_cuda', action='store_true', dest='use_cuda',
                        help="Shall cuda be used (default: False)")
    parser.add_argument('--view', default="", metavar="",
                        help="Print summary about a trained model")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]                    

    example(argv)
    
