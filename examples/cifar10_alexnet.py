#---------------------------------------------------------------
import os, sys 
sys.path.insert(0, os.getcwd()) # enables $ python examples/[EXAMPLE].py
#---------------------------------------------------------------
'''
This is the standard Alexnet example from TensorFlow, implemented in ummon. Note that 
currently PyTorch does not have the truncated normal initialization, so the initial behavior
will be probably different.

Author: M.O.Franz
Copyright (C) 2018 by IOS Konstanz 
'''
import numpy as np
import torch
import torch.nn as nn

import load_cifar10
from ummon import *

# read CIFAR10 data set
X0,y0,X1,y1,X2,y2 = load_cifar10.read("")

# convert to float 64
x0 = X0.astype('float32')
x1 = X1.astype('float32')
x2 = X2.astype('float32')
y0 = np.argmax(y0, axis=1).astype('int64') # pytorch cross entropy expects class label as target
y1 = np.argmax(y1, axis=1).astype('int64')
y2 = np.argmax(y2, axis=1).astype('int64')

# create network (Alexnet similar to TensorFlow example)
cnet = Sequential(
    ('unfla', Unflatten(        [3072],         [3,32,32])),
    ('crop0', Crop(             [3,32,32],      24, 24)),
    ('flip0', RandomFlipLR(     [3,24,24])),
    ('rndbr', RandomBrightness( [3,24,24],      63.0)),
    ('rndco', RandomContrast(   [3,24,24],      0.2, 1.8)),
    ('white', Whiten(           [3,24,24])),
    
    ('conv0', Conv(             [3,24,24],      [64,5,5], init='xavier_normal_', padding=2)),
    ('relu0', nn.ReLU()),
    ('pool0', MaxPool(          [64,24,24],     kernel_size=(3,3), stride=(2,2), padding=1)),
    ('locr0', LRN(              [64,12,12],     9, 1.0, 0.001/9.0, 0.75)),
    
    ('conv1', Conv(             [64,12,12],     [64,5,5], init='xavier_normal_', padding=2)),
    ('relu1', nn.ReLU()),
    ('locr1', LRN(              [64,12,12],     9, 1.0, 0.001/9.0, 0.75)),
    ('pool1', MaxPool(          [64,12,12],     kernel_size=(3,3), stride=(2,2), padding=1)),
    
    ('flatt', Flatten(          [64,6,6])),
    ('line0', Linear(           [64*36],        384, init='xavier_normal_')),
    ('relu2', nn.ReLU()),
    
    ('line1', Linear(           [384],          192, init='xavier_normal_')),
    ('relu3', nn.ReLU()),
    
    ('line2', Linear(           [192],          10, init='xavier_normal_')),
)

cnet.to('cuda')

# loss (size_averaging is numerically unstable)
loss = nn.CrossEntropyLoss(reduction='sum') # inside net: combination of softmax and llh

# optimizer
params_dict = dict(cnet.named_parameters())
params = []
for key, value in params_dict.items():
    if key == 'line0.weight' or key == 'line1.weight':
        params += [{'params':value,'weight_decay':0.004}] # set weight decay only for first two linear layers
    else:
        params += [{'params':value,'weight_decay':0.0}]
mbs = 128                    # batch size
eta = 0.1                    # learning rate
opt = torch.optim.SGD(params, lr=eta/mbs)

# training state
trs = Trainingstate()

with Logger(loglevel=10, logdir='examples/CIFAR10', log_batch_interval=40) as lg:
    
    # scheduler
    scd = StepLR_earlystop(opt, trs, cnet, step_size = 30, nsteps=3, logger=lg, patience=50)
    
    # trainer
    trn = ClassificationTrainer(lg, cnet, loss, opt, trainingstate=trs, scheduler=scd, 
        model_filename='cifar10', combined_training_epochs=5)
    
    # train
    trn.fit((x0,y0,mbs), epochs=90, validation_set=(x2,y2))
    
    # evaluate on test set
    ev = ClassificationAnalyzer.evaluate(cnet, loss, (x1,y1), lg)
    lg.info("Performance on test set: loss={:6.4f}; {:.2f}% correct".format(
        ev["loss"]/y1.shape[0]*mbs,100.0*ev["accuracy"]))

