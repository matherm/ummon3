#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################
'''
This script creates a basic 3-layer linear network with sigmoid activation functions and
30 hidden neurons. On MNIST, the 30-10-network reaches around 96 % correct on the 
validation data (before retraining with the combined training and validation set) in 
maximally 30 epochs using the cross entropy loss function. On the test set, the accuracy 
should be also around 96 %.

Run command:
    
    python mnist1_ummon.py

Author: M.O.Franz
Copyright (C) 2018 by IOS Konstanz 
'''
import torch.nn as nn

import load_mnist
from ummon import *


# read MNIST data set and scale it
X0,y0,Xv,yv,X1,y1 = load_mnist.read([0,1,2,3,4,5,6,7,8,9], path="")
x0 = (1.0/255.0)*X0.astype('float32')
x1 = (1.0/255.0)*X1.astype('float32')
x2 = (1.0/255.0)*Xv.astype('float32')
y0 = y0.astype('float32')
y1 = y1.astype('float32')
y2 = yv.astype('float32')

# network
cnet = Sequential(
    ('line0', Linear( [784],       30, 'xavier_normal')),
    ('sigm0', nn.Sigmoid()),
    ('line1', Linear( [30],        10, 'xavier_normal'))
)

# loss
loss = nn.BCEWithLogitsLoss()

# optimizer
opt = torch.optim.SGD(cnet.parameters(), lr=0.1)

with Logger(logdir='.', log_batch_interval=5000) as lg:
    
    # trainer
    trn = ClassificationTrainer(lg, cnet, loss, opt, precision=np.float32)
    
    # train
    trn.fit((x0,y0,16), epochs=35, validation_set=(x2,y2), eval_interval=5)

