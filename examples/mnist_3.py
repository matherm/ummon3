#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################
'''
This script creates a 2-layer convolutional / 2-layer fully connected network with 
ReLU activation functions, drop-out in the 3rd layer and 512 hidden neurons. This network 
corresponds to network 3 in the ebook by Michael A. Nielsen, "Neural Networks and Deep Learning", 
Determination Press, 2015, but has less hidden neurons and no data augmentation. netw3 
reaches up to 99.26 % on the MNIST validation set before early stopping and retraining. 

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
y0 = np.argmax(y0, axis=1).astype('int64') # pytorch cross entropy expects class label as target
y1 = np.argmax(y1, axis=1).astype('int64')
y2 = np.argmax(yv, axis=1).astype('int64')

# network parameters
mbs = 16                    # batch size
eta = 0.03                  # learning rate
wdecay = 0.1/x0.shape[0]    # weight decay
epochs = 30

# network
cnet = Sequential(
    ('unfla', Unflatten([784],          [1,28,28])),
    ('conv0', Conv(     [1,28,28],      [20,5,5], init='xavier_normal_')),
    ('pool0', MaxPool(  [20,24,24],     (2,2), (2,2))),
    ('relu0', nn.ReLU()),
    ('conv1', Conv(     [20,12,12],     [40,5,5], init='xavier_normal_')),
    ('pool1', MaxPool(  [40,8,8],       (2,2), (2,2))),
    ('relu1', nn.ReLU()),
    ('flatt', Flatten(  [40,4,4])),
    ('line0', Linear(   [640],          516, 'xavier_normal_')),
    ('drop0', Dropout(  [516],          0.5)),
    ('relu2', nn.ReLU()),
    ('line1', Linear(   [516],          10, 'xavier_normal_'))
)
print(cnet)

# loss (size_averaging is numerically unstable)
loss = nn.CrossEntropyLoss(reduction='sum') # inside net: combination of softmax and llh

# optimizer
params_dict = dict(cnet.named_parameters())
params = []
for key, value in params_dict.items():
    if key == 'line0.weight' or key == 'line1.weight':
        params += [{'params':value,'weight_decay':wdecay}] # set weight decay only for linear layers
    else:
        params += [{'params':value,'weight_decay':0.0}]
opt = torch.optim.SGD(params, lr=eta/mbs, weight_decay=wdecay)

# training state
trs = Trainingstate()

with Logger(loglevel=20, logdir='.', log_batch_interval=1700) as lg:
    
    # scheduler
    scd = StepLR_earlystop(opt, trs, cnet, step_size = epochs, nsteps=1, logger=lg, patience=5)
    
    # trainer
    trn = ClassificationTrainer(lg, cnet, loss, opt, trainingstate=trs, scheduler=scd, 
        model_filename='mnist_3', combined_training_epochs=5)
    
    # train
    trn.fit((x0,y0,mbs), epochs=epochs, validation_set=(x2,y2))
    
    # evaluate on test set
    ev = ClassificationAnalyzer.evaluate(cnet, loss, (x1,y1), lg)
    lg.info("Performance on test set: loss={:6.4f}; {:.2f}% correct".format(
        ev["loss"]/y1.shape[0]*mbs,100.0*ev["accuracy"]))

