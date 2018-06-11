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
The script just displays correctly classified examples vs. all examples on the validation 
and test set. 

Author: M.O.Franz
Copyright (C) 2018 by IOS Konstanz 
'''
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

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
y3 = Variable(torch.LongTensor(y1), requires_grad=False)

# network parameters
mbs = 16                    # batch size
eta = 0.03                  # learning rate
wdecay = 0.1/x0.shape[0]    # weight decay
epochs = 30

# network
cnet = Sequential(
    ('unfla', Unflatten([784],          [1,28,28])),
    ('conv0', Conv(     [1,28,28],      [20,5,5], init='xavier_normal')),
    ('pool0', MaxPool(  [20,24,24],     (2,2), (2,2))),
    ('relu0', nn.ReLU()),
    ('conv1', Conv(     [20,12,12],     [40,5,5], init='xavier_normal')),
    ('pool1', MaxPool(  [40,8,8],       (2,2), (2,2))),
    ('relu1', nn.ReLU()),
    ('flatt', Flatten(  [40,4,4])),
    ('line0', Linear(   [640],          516, 'xavier_normal')),
    ('drop0', Dropout(  [516],          0.5)),
    ('relu2', nn.ReLU()),
    ('line1', Linear(   [516],          10, 'xavier_normal'))
)
print(cnet)

# loss (size_averaging is numerically unstable)
loss = nn.CrossEntropyLoss(size_average = False) # inside net: combination of softmax and llh

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
        combined_training_epochs=5)
    
    # train
    trn.fit((x0,y0,mbs), epochs=epochs, validation_set=(x2,y2))

#     
# # create network
# with Network(seed=-1, loglevel=logging.DEBUG, mini_batch_size=mbs) as cnet:
#     unfl0  = Unflatten(  'Unfl',     [784],     cnet,[],         [1,28,28])
#     conv0  = Conv(       'Conv 0',   [1,28,28], cnet,['Unfl'],   [20,5,5], 'xavier_gaussian')
#     pool0  = Pooling(    'Pool 0',   [20,24,24],cnet,['Conv 0'], 'max_pooling', 'valid', 2, 2, 2, 2)
#     act0   = Activation( 'Relu 0',   [20,12,12],cnet,['Pool 0'], 'relu')
#     conv1  = Conv(       'Conv 1',   [20,12,12],cnet,['Relu 0'], [40,5,5], 'xavier_gaussian')
#     pool1  = Pooling(    'Pool 1',   [40,8,8],  cnet,['Conv 1'], 'max_pooling', 'valid', 2, 2, 2, 2)
#     act1   = Activation( 'Relu 1',   [40,4,4],  cnet,['Pool 1'], 'relu')
#     fl0    = Flatten(    'Flatten',  [40,4,4],  cnet,['Relu 1'])
#     lin0   = LinearLayer('Linear 0', [640],     cnet,['Flatten'], 516, 'xavier_gaussian', wdecay)
#     drop   = Dropout(    'Dropout',  [516],     cnet,['Linear 0'],0.5)
#     act2   = Activation( 'Relu 2',   [516],     cnet,['Dropout'], 'relu')
#     lin1   = LinearLayer('Linear 1', [516],     cnet,['Relu 2'],  10, 'xavier_gaussian', wdecay)
#     act3   = Activation( 'Softmax',  [10],      cnet,['Linear 1'],'softmax')
#     loss   = Loss(       'Loss',     [10],      cnet,['Softmax'], 'log_likelihood')
#     opt    = Optimizer(                         cnet, eta, 0.0, 'sgd')
#     cnet.init_network()
#     print(cnet)
#     
#     # train
#     cnet.fit(x0, y0, epochs, True, x2, y2, early_stop=5, report_every=1700)
#     
#     # error on test  set
#     corr = cnet.ncorrect(x1, y1)
#     cnet.log("Error on test set: loss={:6.4f}; {}/{} - {:.2f}% correct".format(cnet.avg_loss(x1, y1), 
#         corr, X1.shape[0], 100.0*corr/x2.shape[0]))
#     
#     # save trained network
#     cnet.save('mnist_3')
#     
# 
