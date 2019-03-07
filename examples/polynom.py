#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys

import logging

#############################################################################################

"""

ummon3 Examples

Polynom

Run command:

    python polynom.py --epochs 1500 --batch_size 3

"""

#
# IMPORTS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
import matplotlib.pyplot as plt
from ummon.logger import Logger
from ummon.trainingstate import Trainingstate
from ummon.supervised import *
from ummon import *
import time

#
# SET inital seed for reproducibility
np.random.seed(1337)
torch.manual_seed(1337)

#
# DEFINE a neural network
class Net(nn.Module):
    # define Layers
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 100)
        self.fc2 = nn.Linear(100, 200)
        self.fc21 = nn.Linear(200, 100)
        self.d1 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(100, 1)

    # connect Layers
    def forward(self, x):
        x = x.view(-1, 1)
        x = F.elu(self.fc1(x))
        #x = self.d1(x)
        x = F.elu(self.fc2(x))
        x = self.d1(x)
        x = F.elu(self.fc21(x))
        x = self.fc3(x)
        return x


def polynom_func(x):
    return 5 * (x ** 5) + 20 * (x ** 4) - 0.1 * (x ** 3) + 20* (x ** 2) + x + 1
    #return (x+4) * (x+2) * (x+1) * (x-1) * (x-3) / 20 + 2



# GENERATE Polynom X and Y values
def generate_training_data(samples=1000, noise=0.0, range=0.0, as_numpy=False):
    data = (np.linspace(-4 - range, 2 + range, num=samples))
    label = (polynom_func(data) + np.random.randn(samples) * noise)
    data = data.reshape((samples, 1)).astype(np.float32)
    label = label.reshape((samples, 1)).astype(np.float32)
    if as_numpy:
        return data, label
    return torch.from_numpy(data), torch.from_numpy(label)


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='ummon3 - example - polynom')
    #
    # TRAINING PARAMS
    parser.add_argument('--epochs', type=int, default=1500, metavar='',
                        help='Amount of epochs for training (default: 1500)')
    parser.add_argument('--batch_size', type=int, default=3, metavar='',
                        help='Batch size for SGD (default: 3)')
    parser.add_argument('--lrate', type=float, default=0.000008, metavar="",
                        help="Learning rate (default: 0.01; elu: 0.000008")
    parser.add_argument('--use_cuda', action='store_true', dest='use_cuda',
                        help="Shall cuda be used (default: False)")
    parser.add_argument('--view', default="", metavar="",
                        help="Print summary about a trained model")
    parser.add_argument('--eval_interval', type=int, default=2000, metavar='',
                        help='Evaluate model in given interval (epochs) (default: 500)')
    parser.add_argument('--samples', type=int, default=100, metavar='',
                        help='Number of Samples in Trainingdata (default: 100)')
    parser.add_argument('--plot', action='store_false', default=True,
                        help='Show a plot with learned Values (default: False)')
    parser.add_argument('--trainingstate', action='store_true', default=False,
                        help='Try to load existing Trainingstate (default: False)')
    parser.add_argument('--noise', type=float, default=1.0, metavar='',
                        help='Adds noise to Training Data (default: 0.0)')
    
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]
    
    argv.use_cuda = torch.cuda.is_available()

    if argv.view is not "":
        ts = Trainingstate(argv.view)
        print(ts.get_summary())

    else:
        # PREPARE TEST DATA
        polynom_data, polynom_label = generate_training_data(samples=argv.samples, noise=argv.noise)
        permut = np.random.permutation(len(polynom_data))
        validation_data, validation_label = polynom_data[permut[:20]], polynom_label[permut[:20]] # num samples = 20
        training_data  , training_label   = polynom_data[permut[20:]], polynom_label[permut[20:]] # num samples = 80
        test_data      , test_label       = generate_training_data(samples=100, noise=5., range=.0)
        
        dataloader_trainingdata = DataLoader(TensorDataset(training_data, training_label), batch_size=argv.batch_size,
                                             shuffle=True, sampler=None,
                                             batch_sampler=None, num_workers=2)

        # CHOOSE MODEL
        model = Net()

        # CHOOSE LOSS-Function
        criterion = nn.MSELoss()

        lr_loss_arr = []

        # INSTANTIATE OPTIMIZER
        optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate)

        # LOAD TRAINING STATE
        ts = Trainingstate()
        if argv.trainingstate:
            try:
                ts = Trainingstate("POLYNOM_best_validation_loss.pth.tar")
            except FileNotFoundError:
                ts = Trainingstate()

        with Logger(logdir='.', log_batch_interval=1000, log_epoch_interval=1, loglevel=1) as lg:

            # CREATE A TRAINER
            my_trainer = SupervisedTrainer(lg,
                                 model,
                                 criterion,
                                 optimizer,
                                 trainingstate = ts,
                                 precision=np.float32,
                                 use_cuda=argv.use_cuda)
            start = time.time()
            print("Start")
            # START TRAINING
            my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                            epochs=argv.epochs,
                                            validation_set=TensorDataset(validation_data, validation_label))
            end = time.time()
            print("Zeit: ", end - start)
            results_test  = Predictor.predict(model, test_data, logger=lg)
            results_train = Predictor.predict(model, training_data, logger=lg)
            results_valid = Predictor.predict(model, validation_data, logger=lg)
            
            
            print("MSE (test_data)",       np.mean((results_test  - test_label).numpy()**2))
            print("MSE (training_data)",   np.mean((results_train - training_label).numpy()**2))
            print("MSE (validation_data)", np.mean((results_valid - validation_label).numpy()**2))
            
            std_valid = (results_valid - validation_label).std()
            upper_confidence = results_valid + std_valid
            lower_confidence = results_valid - std_valid
            
            #PLOT results
            if argv.plot:
                #x, y = generate_training_data(samples=int(100), as_numpy=True)
                plt.figure("ummon3 Example Polynom:")
                plt.plot(validation_data.numpy(), results_valid.numpy(), "o", label="Net Output")
                plt.plot(validation_data.numpy(), upper_confidence.numpy(), "m+", label="Upper Confidence")
                plt.plot(validation_data.numpy(), lower_confidence.numpy(), "k+", label="Lower Confidence")
                if argv.noise > 0.0:
                    plt.plot(training_data.numpy(), training_label.numpy(), '*', label="Training Data")
                plt.plot(polynom_data.numpy(), polynom_label.numpy(), "-", label="Function(x)")
                plt.legend()
                plt.show()