#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys

import logging

sys.path.insert(0, '../../ummon3')  # for python basicusage.py
sys.path.insert(0, '../ummon3')  # for python examples/basicusage.py
#############################################################################################

"""

ummon3 Examples

Sinus

Run command:

    python sinus.py --epochs 1 --batch_size 40

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
from ummon.trainer import Trainer
from ummon.logger import Logger
from ummon.trainingstate import Trainingstate
from ummon.supervised import *

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
        self.fc1 = nn.Linear(1, 30)
        self.fc2 = nn.Linear(30, 90)
        self.fc21 = nn.Linear(90, 30)
        self.fc3 = nn.Linear(30, 1)

    # connect Layers
    def forward(self, x):
        x = x.view(-1, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc21(x))
        x = self.fc3(x)
        return x.cpu()

#
# GENERATE Sinus X and Y values
def generate_training_data(samples=100, noise=0.0, periods=1, as_numpy=False):
    data = (np.linspace(0, 2 * periods * math.pi, num=samples))
    label = (np.sin(data) + np.random.randn(samples) * noise)
    data = data.reshape((samples, 1)).astype(np.float32)
    label = label.reshape((samples, 1)).astype(np.float32)
    if as_numpy:
        return data, label
    return torch.from_numpy(data), torch.from_numpy(label)


if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description='ummon3 - example - sinus')
    #
    # TRAINING PARAMS
    parser.add_argument('--epochs', type=int, default=1500, metavar='',
                        help='Amount of epochs for training (default: 1500)')
    parser.add_argument('--batch_size', type=int, default=40, metavar='',
                        help='Batch size for SGD (default: 40)')
    parser.add_argument('--lrate', type=float, default=0.01, metavar="",
                        help="Learning rate (default: 0.01")
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
    parser.add_argument('--periods', type=float, default=1.0, metavar='',
                        help='Number of Sinus-Periods (default: 1.0)')
    parser.add_argument('--noise', type=float, default=0.0, metavar='',
                        help='Adds noise to Training Data (default: 0.0)')
    
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]


    if argv.view is not "":
        ts = Trainingstate(argv.view)
        print(ts.get_summary())

    else:
        # PREPARE TEST DATA
        sinus_data, sinus_label = generate_training_data(samples=argv.samples, noise=argv.noise, periods=argv.periods)
        dataloader_trainingdata = DataLoader(TensorDataset(sinus_data, sinus_label), batch_size=argv.batch_size,
                                             shuffle=True, sampler=None,
                                             batch_sampler=None, num_workers=2)

        tes, test = generate_training_data(samples=99, noise=0, periods=argv.periods)

        # CHOOSE MODEL
        model = Net()

        # CHOOSE LOSS-Function
        criterion = nn.MSELoss()

        # INSTANTIATE OPTIMIZER
        optimizer = torch.optim.SGD(model.parameters(), lr=argv.lrate)

        # LOAD TRAINING STATE
        ts = None
        if argv.trainingstate:
            try:
                ts = Trainingstate("SINUS_best_validation_loss.pth.tar")
            except FileNotFoundError:
                ts = None

        with Logger(logdir='.', log_batch_interval=1000, log_epoch_interval=100, loglevel=1) as lg:

            # CREATE A TRAINER
            my_trainer = Trainer(lg,
                                 model,
                                 criterion,
                                 optimizer,
                                 model_filename="SINUS",
                                 precision=np.float32,
                                 use_cuda=argv.use_cuda)

            # START TRAINING
            trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                            epochs=argv.epochs,
                                            validation_set=TensorDataset(tes, test),
                                            eval_interval=argv.eval_interval,
                                            trainingstate=ts)

            results = Analyzer.inference(model, dataset=TensorDataset(tes, test), logger=lg)

            #PLOT results
            if argv.plot:
                x, y = generate_training_data(samples=int(100 * argv.periods), periods=argv.periods, as_numpy=True)
                plt.figure("ummon3 Example Sinus:")
                plt.plot(tes.numpy(), results.numpy(), "o", label="Net Output")
                if argv.noise > 0.0:
                    plt.plot(sinus_data.numpy(), sinus_label.numpy(), '*', label="Training Data")
                plt.plot(x, y, "-", label="Sinus(x)")
                plt.legend()
                plt.show()
