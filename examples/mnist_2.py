#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  # for python basicusage.py
sys.path.insert(0,'../ummon3')     # for python examples/basicusage.py
#############################################################################################
'''
This script creates a 3-layer linear network with sigmoid activation functions and 100 
hidden neurons. This network corresponds to network 2 in the ebook by Michael A. Nielsen, 
"Neural Networks and Deep Learning", Determination Press, 2015. The improved netw2 
typically reaches up to 97.8-98.0 % on the MNIST validation set before early stopping. 
The script just displays correctly classified examples vs. all examples on the validation 
and test set. On the MNIST test set, the net reaches around 97.9 % correct.

Author: M.O.Franz
Copyright (C) 2018 by IOS Konstanz 
'''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

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
x3 = Variable(torch.FloatTensor(x1), requires_grad=False)
        
# network parameters
mbs = 16                        # batch size
eta = 0.1                       # learning rate
no_hidden = 100                 # number of hidden neurons 
wdecay = eta*5.0/x0.shape[0]    # weight decay
epochs = 60                 

# network
net = Sequential(
    ('line0', Linear( [784], no_hidden, 'xavier_normal')),
    ('sigm0', nn.Sigmoid()),
    ('line1', Linear( [no_hidden], 10,  'xavier_normal'))
)

# loss (size_averaging is numerically unstable)
loss = nn.CrossEntropyLoss(size_average = False) # inside net: combination of softmax and llh

# optimizer
opt = torch.optim.SGD(net.parameters(), lr=eta/mbs, weight_decay=wdecay)

with Logger(loglevel=20, logdir='.', log_batch_interval=5000) as lg:
    
    # trainer
    trn = ClassificationTrainer(lg, net, loss, opt, combined_training_epochs=5)
    
    # train
    trn.fit((x0,y0,mbs), epochs=epochs, validation_set=(x2,y2))
    
    # predict on test set
    y1_pred = net(x3)
    
    # evaluate
    correct = np.sum(y1 == np.argmax(y1_pred.data.numpy(), axis=1))
    lg.info("Performance on test set: {:.2f}% correct".format(100.0*correct/y1.shape[0]))

