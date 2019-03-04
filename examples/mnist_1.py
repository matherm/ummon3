#---------------------------------------------------------------
import os, sys 
sys.path.insert(0, os.getcwd()) # enables $ python examples/[EXAMPLE].py
#---------------------------------------------------------------
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
import numpy as np
import torch
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
net = Sequential(
    ('line0', Linear( [784], 30, 'xavier_normal_')),
    ('sigm0', nn.Sigmoid()),
    ('line1', Linear( [30],  10, 'xavier_normal_'))
)

# loss (size_averaging is numerically unstable)
loss = nn.BCEWithLogitsLoss(reduction='sum')

# optimizer
mbs = 16
opt = torch.optim.SGD(net.parameters(), lr=1/mbs)

# training state
trs = Trainingstate()

with Logger(loglevel=20, logdir='.', log_batch_interval=5000) as lg:
    
    # scheduler
    scd = StepLR_earlystop(opt, trs, net, step_size = 35, nsteps=2, logger=lg, gamma=0.1, patience=5)
    
    # trainer
    trn = ClassificationTrainer(lg, net, loss, opt, trainingstate=trs, scheduler=scd, 
        model_filename='mnist1', combined_training_epochs=5)
    
    # train
    trn.fit((x0,y0,mbs), epochs=70, validation_set=(x2,y2))
    
    
    # evaluate on test set
    ev = ClassificationAnalyzer.evaluate(net, loss, (x1,y1), lg)
    lg.info("Performance on test set: loss={:6.4f}; {:.2f}% correct".format(
        ev["loss"]/y1.shape[0]*mbs,100.0*ev["accuracy"]))

