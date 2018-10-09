#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import unittest
import logging
import os
import numpy as np
from scipy.signal import convolve2d, correlate2d
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import ummon.utils as uu
from ummon import *

# set fixed seed for reproducible results
torch.manual_seed(4)
np.random.seed(4)

def sigmoid(z):
    z = np.clip(z, log(np.finfo(np.float64).tiny), log(np.finfo(np.float64).max) - 1.0)
    return 1.0/(1.0+np.exp(-z))

class TestUmmon(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestUmmon, self).__init__(*args, **kwargs)
        
        # BACKUP files
        backup_dir = "_backup"
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension) or file.endswith(".log"):
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                os.rename(os.path.join(dir,file), os.path.join(backup_dir,file))
        
    # test fully connected layer
    def test_predict(self):
        print('\n')
        
        # create net
        cnet = Sequential(
            ('line0', Linear([5], 7, 'xavier_uniform_', 0.001)),
            ('sigm0', nn.Sigmoid())
        )
        print(cnet)
        
        # check weight matrix
        w = cnet.line0.w
        b = cnet.line0.b
        print('Weight matrix:')
        print(w)
        print('bias:')
        print(b)
        
        # predict
        x0 = np.random.randn(1,5).astype('float32')
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y1 = cnet(x1)
        y1 = y1.data.numpy()
        print('Predictions:')
        print(y1)
        assert y1.shape[1] == 7
        
        # check
        x0 = x0.reshape((5, 1))
        y2 = sigmoid(np.dot(w, x0) + b.reshape((7, 1)))
        print('Reference predictions:')
        print(y2.transpose())
        assert np.allclose(y1, y2.transpose(), 0, 1e-5)    
    
    
    # test loss
    def test_loss(self):
        print('\n')
        
        # test data
        x0 = np.random.rand(6,5).astype('float32')
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y0 = np.zeros((6,5), dtype=np.float32)
        y0[:,2] = np.ones(6, dtype=np.float32) # log likelihood works only for one hot coded outputs
        y1 = Variable(torch.FloatTensor(y0), requires_grad=False)
        
        # test MSE loss
        loss = nn.MSELoss(reduction='sum')
        mse = loss(x1, y1).data.numpy()
        print('MSE:     ', mse)
        mse_true = ((x0 - y0)**2).sum() # pyTorch divides by n_dim instead of 2
        print('True MSE:', mse_true)
        assert np.allclose(mse, mse_true, 0, 1e-3)
        
        # test log likelihood loss function
        y1 = Variable(torch.LongTensor(2*y0[:,2]), requires_grad=False) # pytorch takes class index, not one-hot coding
        loss = nn.NLLLoss(reduction='sum')
        LL = loss(x1, y1).data.numpy()
        print('LL:     ', LL)
        # should be LL_true = (-y0*np.nan_to_num(np.log(x0))).sum(axis=1).mean(), but pytorch expects x1 to be already log'd by Log Softmax 
        LL_true = (-y0*x0).sum()
        print('True LL:', LL_true)
        assert np.allclose(LL, LL_true, 0, 1e-3)
        
        # test pytorch cross entropy loss function (=logsoftmax + NLL)
        loss = nn.CrossEntropyLoss(reduction='sum')
        ce = loss(x1, y1).data.numpy()
        print('CE:      ', ce)
        # pytorch CE is combination of log softmax and log likelihood
        ce_true = (-x0[:,2] + np.log(np.exp(x0).sum(axis=1))).sum()
        print('True CE: ', ce_true)
        assert np.allclose(ce, ce_true, 0, 1e-3)
        
        # test binary cross entropy
        loss = nn.BCELoss(reduction='sum')
        y1 = Variable(torch.FloatTensor(y0), requires_grad=False)
        bce = loss(x1, y1).data.numpy()
        print('BCE:     ', bce)
        bce_true = (np.nan_to_num(-y0*np.log(x0)-(1-y0)*np.log(1-x0))).sum()
        print('True BCE:', bce_true) # pytorch takes mean across dimensions instead of sum
        assert np.allclose(bce, bce_true, 0, 1e-3)
        
        # test pytorch combined sigmoid and bce
        loss = nn.BCEWithLogitsLoss(reduction='sum')
        bce = loss(x1, y1).data.numpy()
        print('BCEL:    ', bce)
        bce_true = (np.nan_to_num(-y0*np.log(sigmoid(x0))-(1-y0)*np.log(1-sigmoid(x0)))).sum()
        print('TrueBCEL:', bce_true)
        assert np.allclose(bce, bce_true, 0, 1e-3)
        
        # test embedding hinge loss function of pytorch (this is not the true Hinge loss!)
        x0 = np.random.rand(6,1).astype('float32')
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y0 = np.ones(((6,1)), dtype=np.float32)
        y0[3:] = -1
        y1 = Variable(torch.FloatTensor(y0), requires_grad=False)
        loss = nn.HingeEmbeddingLoss()
        hinge = loss(x1, y1).data.numpy()
        print('HingeE:  ', hinge)
        hinge_true = (x0[:3].sum() + np.maximum(0, 1 - x0[3:]).sum())/6.0
        print('TrueHinE:', hinge_true)
        assert np.allclose(hinge, hinge_true, 0, 1e-3)
        
        # test true standard hinge loss
        loss = nn.MarginRankingLoss(reduction='sum', margin=1.0)
        dummy = torch.FloatTensor(6,1).zero_()
        # dummy variable must have same size as x1, but must be 0
        hinge = loss(x1, dummy, y1).data.numpy()
        print('Hinge:   ', hinge)
        hinge_true = (np.maximum(0, 1 - x0*y0)).sum()
        print('True Hin:', hinge_true)
        assert np.allclose(hinge, hinge_true, 0, 1e-3)
    
    
    # check gradient
    def test_gradient(self):
        
        def L(w, b, x, y, lossfunc, act, tck=None):
            
            # linear layer
            x1 = x.transpose()
            y1 = (np.dot(w, x1) + b.reshape((5, 1))).transpose()
            
            # activation function
            if act == 'sigmoid':
                y2 = sigmoid(y1)
            elif act == 'logsoftmax':
                y2 = np.exp(y1)
                denom = np.tile(y2.sum(axis=1).reshape((4,1)), (1,5))
                y2 = np.log(y2/denom)
            elif act == 'bspline':
                y2 = sp.splev(y1, tck)
            
            # loss
            if lossfunc == 'mse':
                lo =  ((y - y2)**2).sum()
            elif lossfunc == 'cross_entropy': # binary cross entropy
                lo = (np.nan_to_num(-y*np.log(y2)-(1-y)*np.log(1-y2))).sum()
            elif lossfunc == 'log_likelihood': # log likelihood
                lo = (-y*y2).sum()
            
            return lo
        
        def check_grad_activation(lossfunc, act):
            
            print('\n')
            print("Loss function: {}, Activation: {}".format(lossfunc, act))
            batch = 4
            eta = 0.5
            
            # activation
            if act == 'sigmoid':
                nonl = nn.Sigmoid()
            elif act == 'logsoftmax':
                nonl = nn.LogSoftmax(dim=1)
            
            # net
            cnet = Sequential(
                ('line0', Linear([3], 5, 'xavier_normal_')),
                ('nonl0', nonl)
            )
            print(cnet)
            
            # loss
            if lossfunc == 'mse':
                loss = nn.MSELoss(reduction='sum')
            elif lossfunc == 'cross_entropy':
                loss = nn.BCELoss(reduction='sum')
            elif lossfunc == 'log_likelihood': 
                loss = nn.NLLLoss(reduction='sum')
            
            # get weights
            w0 = cnet.line0.w
            b0 = cnet.line0.b
            
            # training data
            x0 = 0.1*np.random.rand(batch, 3).astype('float32') + 0.5
            x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
            y0 = np.zeros((batch, 5), dtype=np.float32)
            y0[:,2] = np.ones((batch), dtype=np.float32) # log likelihood works only for one hot coded outputs
            if lossfunc == 'log_likelihood': 
                y1 = Variable(torch.LongTensor(2*y0[:,2]), requires_grad=False) # pytorch takes class index, not one-hot coding
            else:
                y1 = Variable(torch.FloatTensor(y0), requires_grad=False)
            
            # torch gradient
            y2 = cnet(x1)
            lo = loss(y2, y1)
            lo.backward()
            print('Analytically computed gradient of w:')
            dW0 = cnet.line0.weight.grad.data.numpy()
            print(dW0)
            
            # numerical gradient
            eps = 0.001 # small gradient step
            dW = np.zeros((5,3), dtype=np.float32)
            for r in range(5):
                for c in range(3):
                    w0[r,c] += eps
                    L1 = L(w0, b0, x0, y0, lossfunc, act)
                    w0[r,c] -= 2.0*eps
                    L2 = L(w0, b0, x0, y0, lossfunc, act)
                    w0[r,c] += eps
                    dW[r,c] = (L1 - L2)/(2*eps)
            print('Numerically computed gradient of w:')
            print(dW)
            assert np.allclose(dW, dW0, 1e-1)
            
            print('Analytically computed gradient of b:')
            db0 = cnet.line0.bias.grad.data.numpy()
            print(db0)
            
            db = np.zeros((5,1), dtype=np.float32)
            for r in range(5):
                b0[r] += eps
                L1 = L(w0, b0, x0, y0, lossfunc, act)
                b0[r] -= 2.0*eps
                L2 = L(w0, b0, x0, y0, lossfunc, act)
                b0[r] += eps
                db[r] = (L1 - L2)/(2*eps)
            print('Numerically computed gradient of b:')
            db = db.reshape((5,))
            print(db)
            assert np.allclose(db, db0, 1e-1)
        
        check_grad_activation('mse', 'sigmoid')
        check_grad_activation('cross_entropy', 'sigmoid')
        check_grad_activation('log_likelihood', 'logsoftmax')
    
    
    # test logger
    def test_logger(self):
        print('\n')
        
        with Logger('ummon', 10) as lg: # create net
            lg.debug('Test debugging output.')
            lg.info('Test info.')
            lg.warn('Test warning!')
            try:
                lg.error('Test error!', ValueError)
            except ValueError:
                print("Only a test - no worries ...")
    
    
    # test training
    def test_train(self):
        print('\n')
        
        # create net
        batch = 4
        eta = 0.1/batch
        cnet = Sequential(
            ('line0', Linear([3], 2, 'xavier_normal_'))
        )
        loss = nn.BCEWithLogitsLoss(reduction='sum')
        opt = torch.optim.SGD(cnet.parameters(), lr=eta)
        w = cnet.line0.w
        b = cnet.line0.b.copy()
        b = b.reshape((2, 1))
        
        # training data
        x0 = np.zeros((2*batch, 3), dtype=np.float32)
        y0 = np.zeros((2*batch, 2), dtype=np.float32)
        x0[:batch,:] = np.random.randn(batch, 3).astype('float32') + 1
        x0[batch:,:] = np.random.randn(batch, 3).astype('float32') - 1
        y0[:batch,0] = np.ones((batch), dtype=np.float32)
        y0[batch:,1] = np.ones((batch), dtype=np.float32)
        
        for i in range(2):
            
            # compute reference forward path
            x1 = x0[i*batch:(i+1)*batch,:].transpose()
            y1 = (np.dot(w, x1) + np.tile(b, (1, batch))).transpose()
            y2 = sigmoid(y1)
            
            # reference backward path
            dL = 0.5*(y2 - y0[i*batch:(i+1)*batch,:])
            db = dL.sum(axis=0).reshape(2,1)
            b = b - eta*db
            dW = np.dot(dL.transpose(), x0[i*batch:(i+1)*batch,:])
            w = w - eta*dW
        
        print('Ref. weight matrix:')
        print(w)
        print('Ref. bias:')
        print(b.T[0])
        
        # fit
        with Logger(logdir='', loglevel=logging.ERROR) as lg:
            trn = ClassificationTrainer(lg, cnet, loss, opt)
            trn.fit((x0, y0, batch), 1, (x0, y0))
        
        # check results
        w0 = cnet.line0.w
        b0 = cnet.line0.b
        print('Weight matrix:')
        print(w0)
        print('bias:')
        print(b0)
        assert np.allclose(w, w0, 0, 1e-1)
        assert np.allclose(b, b0, 0, 1e-1)
    
    
    # test flatten
    def test_flatten(self):
        print('\n')
        
        # create net 
        batch = 5
        cnet = Sequential(
            ('sigm0', nn.Sigmoid()),
            ('flatt', Flatten([1,3,2]))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch, 1, 3, 2).astype('float32')
        
        # compute reference forward path
        y3 = sigmoid(x0)
        y4 = np.reshape(y3, (1,1,batch,6))
        
        # predict and check Flatten
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y1 = cnet(x1)
        y = y1.data.numpy()        
        print('Predictions Flatten:')
        print(y)
        print('Reference predictions:')
        print(y4)
        assert np.allclose(y, y4, 0, 1e-5)
    
    
    # test unflatten
    def test_unflatten(self):
        print('\n')
        
        # create net 
        batch = 5
        cnet = Sequential(
            ('unfla', Unflatten([6], [1,3,2])),
            ('sigm0', nn.Sigmoid())
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch, 6).astype('float32')
        
        # compute reference forward path
        y2 = np.reshape(x0, (batch,1,3,2))
        y3 = sigmoid(y2)
        
        # predict and check unflatten
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y1 = cnet(x1)
        y = y1.data.numpy()        
        print('Predictions Unflatten:')
        print(y)
        print('Reference predictions:')
        print(y3)
        assert np.allclose(y, y3, 0, 1e-5)
    
    
    # test convolutional layer valid padding
    def test_conv_valid(self):
        print('\n')
        
        # create net
        cnet = Sequential(
            ('unfla', Unflatten([25], [1,5,5])),
            ('conv0', Conv([1,5,5], [2,3,3]))
        )
        print(cnet)
        
        # check weight matrix size
        w = cnet.conv0.w
        b = cnet.conv0.b
        print('Weight tensor:')
        print(w)
        print('bias:')
        print(b)
        assert w.shape == (2,1,3,3)
        assert b.shape == (2,)
        
        # test input
        x0 = np.zeros((2,25), dtype=np.float32)
        x0[0,12] = 1.0
        x0[1,12] = 3.0
        y0 = np.random.randn(2, 18).astype('float32')
        y1 = np.zeros((2,2,3,3), dtype=np.float32)
        
        # predict and check cross-correlation
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x1)
        y2 = y2.data.numpy()
        print('Predictions Conv for delta input:')
        print(y2)
        print('Reference predictions:')
        y1[0,0,:,:] = np.flipud(np.fliplr(w[0,0,:,:])) + b[0]
        y1[0,1,:,:] = np.flipud(np.fliplr(w[1,0,:,:])) + b[1]
        y1[1,0,:,:] = np.flipud(np.fliplr(3*w[0,0,:,:])) + b[0]
        y1[1,1,:,:] = np.flipud(np.fliplr(3*w[1,0,:,:])) + b[1]
        print(y1)
        assert np.allclose(y1, y2, 0, 1e-5)
    
    
    # test convolutional layer gradient
    def test_conv_valid_gradient(self):
        print('\n')
        
        # create net
        eta = 0.1
        cnet = Sequential(
            ('unfla', Unflatten([25], [1,5,5])),
            ('conv0', Conv([1,5,5], [2,3,3])),
            ('flatt', Flatten([2,3,3]))
        )
        print(cnet)
        loss = nn.MSELoss(reduction='sum')
        opt = torch.optim.SGD(cnet.parameters(), lr=eta)
        
        # weight matrix
        w = cnet.conv0.w
        b = cnet.conv0.b
        
        # test input
        x0 = np.zeros((2,25), dtype=np.float32)
        x0[0,12] = 1.0
        x0[1,12] = 3.0
        y0 = np.random.randn(2, 18).astype('float32')
        
        # check update
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y1 = Variable(torch.FloatTensor(y0), requires_grad=False)
        y2 = cnet(x1)
        lo = loss(y2, y1)
        lo.backward() # torch gradient
        print('Analytically computed gradient of w:')
        dW0 = cnet.conv0.weight.grad.data.numpy()
        print(dW0)
        
        # reference forward path
        y3 = np.zeros((2,2,3,3), dtype=np.float32)
        y3[0,0,:,:] = np.flipud(np.fliplr(w[0,0,:,:])) + b[0]
        y3[0,1,:,:] = np.flipud(np.fliplr(w[1,0,:,:])) + b[1]
        y3[1,0,:,:] = np.flipud(np.fliplr(3*w[0,0,:,:])) + b[0]
        y3[1,1,:,:] = np.flipud(np.fliplr(3*w[1,0,:,:])) + b[1]
        
        # reference backward path
        y3 = np.reshape(y3, (1,1,2,18))
        dL = 2.0*(y3 - y0)
        dL1 = np.reshape(dL, (2,2,3,3))
        dL2 = np.zeros((2,2,7,7), dtype=np.float32)
        dL2[0,0,2:5,2:5] = dL1[0,0,:,:]
        dL2[0,1,2:5,2:5] = dL1[0,1,:,:]
        dL2[1,0,2:5,2:5] = dL1[1,0,:,:]
        dL2[1,1,2:5,2:5] = dL1[1,1,:,:]
        
        # reference gradient of w
        x3 = np.reshape(x0, (2,1,5,5))
        dW1 = correlate2d(x3[0,0,:,:], dL1[0,0,:,:], 'valid')
        dW2 = correlate2d(x3[0,0,:,:], dL1[0,1,:,:], 'valid')
        dW1 += correlate2d(x3[1,0,:,:], dL1[1,0,:,:], 'valid')
        dW2 += correlate2d(x3[1,0,:,:], dL1[1,1,:,:], 'valid')
        print('True weight gradient:')
        print(dW1)
        print(dW2)
        assert np.allclose(dW0[0,0,:,:], dW1, 0, 1e-5)
        assert np.allclose(dW0[1,0,:,:], dW2, 0, 1e-5)
        
        print('Analytically computed gradient of b:')
        db0 = cnet.conv0.bias.grad.data.numpy()
        print(db0.transpose())
        
        # reference gradient of b
        db1 = np.array([dL1[0,0,:,:].sum(), dL1[0,1,:,:].sum()])
        db1 += np.array([dL1[1,0,:,:].sum(), dL1[1,1,:,:].sum()])
        print('True bias gradient:')
        print(db1)
        assert np.allclose(db0.transpose(), db1, 0, 1e-5)
    
    
    # test max pooling inside valid region
    def test_max_pooling_valid(self):
        
        batch = 1    
        print('\n')
        cnet = Sequential(
            ('unfla', Unflatten([35], [1,5,7])),
            ('pool0', MaxPool([1,5,7], (2,3), (2,3)))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,35).astype('float32')
        
        # compute reference forward path
        x1 = np.reshape(x0, (batch,1,5,7))
        print('Input:')
        print(x1)
        y1 = np.zeros((2,2), dtype=np.float32)
        y1[0,0] = x1[0,0,:2,:3].max()
        y1[0,1] = x1[0,0,:2,3:6].max()
        y1[1,0] = x1[0,0,2:4,:3].max()
        y1[1,1] = x1[0,0,2:4,3:6].max()
        
        # predict and check
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x2)
        y2 = y2.data.numpy()
        print('Predictions max pooling:')
        print(y2)
        print('Reference predictions:')
        print(y1)
        assert np.allclose(y2, y1, 0, 1e-5)
    
    
    # test average pooling inside valid region
    def test_avg_pooling_valid(self):
        
        batch = 1    
        print('\n')
        cnet = Sequential(
            ('unfla', Unflatten([35], [1,5,7])),
            ('pool0', AvgPool([1,5,7], (2,3), (2,3)))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,35).astype('float32')
        
        # compute reference forward path
        x1 = np.reshape(x0, (batch,1,5,7))
        print('Input:')
        print(x1)
        y1 = np.zeros((2,2), dtype=np.float32)
        y1 = np.zeros((2, 2), dtype=np.float32)
        y1[0, 0] = x1[0, 0, :2, :3].mean()
        y1[0, 1] = x1[0, 0, :2, 3:6].mean()
        y1[1, 0] = x1[0, 0, 2:4, :3].mean()
        y1[1, 1] = x1[0, 0, 2:4, 3:6].mean()
        
        # predict and check
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x2)
        y2 = y2.data.numpy()
        print('Predictions max pooling:')
        print(y2)
        print('Reference predictions:')
        print(y1)
        assert np.allclose(y2, y1, 0, 1e-5)
    
    
    # test RELU nonlinearity
    def test_relu(self):
        print('\n')
        cnet = Sequential(
            ('relu0', nn.ReLU())
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(2,2,2,2).astype('float32')
        print('Input:')
        print(x0.flatten())
        
        # predict
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x1)
        y2 = y2.data.numpy()
        print('Predictions ReLu:')
        print(y2.flatten())
        
        # reference forward path
        y1 = np.zeros((2,2,2,2), dtype=np.float32)
        y1[x0 > 0] = x0[x0 > 0]
        print('Reference predictions:')
        print(y1.flatten())
        assert np.allclose(y2, y1, 0, 1e-5)
    
    
    # test dropout layer
    def test_dropout(self):
        print('\n')
        cnet = Sequential(
            ('drop0', Dropout(5, 0.5))
        )
        print(cnet)
        
        # predict
        x0 = np.random.randn(5).astype('float32')
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x1)
        y2 = y2.data.numpy()
        print('Input:')
        print(x0)
        print('Output training:')
        print(y2)
        assert len(x0) == len(y2)
        assert (y2[y2 != 0] == 2.0*x0[y2 != 0]).all()
        cnet.eval()
        y2 = cnet(x1)
        y2 = y2.data.numpy()
        print('Output testing:')
        print(y2)
        assert np.allclose(y2, x0, 0, 1e-5)
    
    
    # test get_max_inputs
    def test_get_max_inputs(self):
        print('\n')
        cnet = Sequential(
            ('unfla', Unflatten([25], [1,5,5])),
            ('conv0', Conv([1,5,5], [2,3,3])),
            ('relu0', nn.ReLU())
        )
        print(cnet)    
        
        # test input
        x0 = np.random.randn(2,25).astype('float32')
        x1 = np.reshape(x0, (2,1,5,5))
        fmap = 1
        print('Unflattened input:')
        print(x1)
        
        # predict and find max activation
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y0 = cnet(x2)
        y0 = y0.data.numpy()
        print('Feature maps:')
        fmap = 1
        y0 = y0[:,fmap,:,:]
        print(y0)
        idx = np.argmax(y0, axis=None)
        multi_idx = np.unravel_index(idx, y0.shape)
        print('Maximum: ', y0.max(), ' at z=', multi_idx)
        
        # get associated unflattened input
        inp = x1[multi_idx[0],0,multi_idx[1]:multi_idx[1]+3,multi_idx[2]:multi_idx[2]+3]
        print('Maximally activating patch:')
        print(inp)
        
        vis = Visualizer(cnet)
        print('Available nonlinearities:')
        print(vis._act_funcs)
        
        y = vis.get_max_inputs('conv0', fmap, 3, x0)
        print('Method output:')
        print(y)
        assert np.allclose(y[0,0,:,:], inp, 0, 1e-5)
    
    
    def test_saliency(self):
        print('\n')
        cnet = Sequential(
            ('unfla', Unflatten([25], [1,5,5])),
            ('conv0', Conv([1,5,5], [2,3,3])),
            ('relu0', nn.ReLU())
        )
        print(cnet)    
        
        # test input
        x0 = np.random.randn(2,25).astype('float32')
        x1 = np.reshape(x0, (2,1,5,5))
        fmap = 1
        print('Unflattened input:')
        print(x1)
        
        # predict and find max activation
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y0 = cnet(x2)
        y0 = y0.data.numpy()
        print('Feature maps:')
        fmap = 1
        y0 = y0[:,fmap,:,:]
        print(y0)
        idx = np.argmax(y0, axis=None)
        multi_idx = np.unravel_index(idx, y0.shape)
        print('Maximum: ', y0.max(), ' at z=', multi_idx)
        
        # get maximally activating patches
        vis = Visualizer(cnet)
        y = vis.saliency('relu0', fmap, 3, x0)
        print('Saliency map (gradient) for first maximum:')
        print(y[0,:,:,:])
        
        # read out patch from input image
        y1 = y[0,0,multi_idx[1]:multi_idx[1]+3, multi_idx[2]:multi_idx[2]+3]
        
        # gradient must be one of the weight matrices of the conv layer
        w = cnet.conv0.w
        print('Weight matrix:')
        print(w)
        assert (np.allclose(y1, w[0,0,:,:], 0, 1e-5) or np.allclose(y1, w[1,0,:,:], 0, 1e-5))
        
        y2, y3 = vis.signed_saliency(y[0,:,:,:])
        print("Positive saliency:")
        print(y2)
        print("Negative saliency:")
        print(y3)
    
    
    # test convolutional layer with half padding
    def test_half_padding(self):
        print('\n')
        
        # create net
        cnet = Sequential(
            ('unfla', Unflatten([25], [1,5,5])),
            ('conv0', Conv([1,5,5], [2,3,3], padding=1))
        )
        print(cnet)
        
        # check weight matrix size
        w = cnet.conv0.w
        b = cnet.conv0.b
        print('Weight tensor:')
        print(w)
        print('bias:')
        print(b)
        assert w.shape == (2,1,3,3)
        assert b.shape == (2,)
        
        # test input
        x0 = np.zeros((2,25), dtype=np.float32)
        x0[0,12] = 1.0
        x0[1,12] = 3.0
        y0 = np.random.randn(2, 50).astype('float32')
        y1 = np.zeros((2,2,5,5), dtype=np.float32)
        
        # predict and check cross-correlation
        x1 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x1)
        y2 = y2.data.numpy()
        print('Predictions Conv for delta input:')
        print(y2)
        print('Reference predictions:')
        y1[0,0,1:-1,1:-1] += np.flipud(np.fliplr(w[0,0,:,:]))
        y1[0,1,1:-1,1:-1] += np.flipud(np.fliplr(w[1,0,:,:]))
        y1[1,0,1:-1,1:-1] += np.flipud(np.fliplr(3*w[0,0,:,:]))
        y1[1,1,1:-1,1:-1] += np.flipud(np.fliplr(3*w[1,0,:,:]))
        print(y1)
        assert np.allclose(y1, y2, 0, 1e-5)
    
    
    # test convolutional layer with half padding gradient
    def test_half_padding_gradient(self):
        print('\n')
    
        def half(img):
            out_img = np.zeros((img.shape[0]+2, img.shape[1]+2), dtype=np.float32)
            out_img[1:-1,1:-1] = img
            return out_img
        
        # create net
        eta = 0.1
        cnet = Sequential(
            ('unfla', Unflatten([25], [1,5,5])),
            ('conv0', Conv([1,5,5], [2,3,3], padding=1)),
            ('flatt', Flatten([2,5,5]))
        )
        print(cnet)
        loss = nn.MSELoss(reduction='sum')
        opt = torch.optim.SGD(cnet.parameters(), lr=eta)
        
        # weights
        w = cnet.conv0.w
        b = cnet.conv0.b
        print('Weight tensor:')
        print(w)
        print('bias:')
        print(b)
        
        # test input
        x0 = np.zeros((2,25), dtype=np.float32)
        x0[0,12] = 1.0
        x0[1,12] = 3.0
        y0 = np.random.randn(2, 50).astype('float32')
        y1 = np.zeros((2,2,5,5), dtype=np.float32)
        
        # forward path
        y1[0,0,1:-1,1:-1] += np.flipud(np.fliplr(w[0,0,:,:]))
        y1[0,1,1:-1,1:-1] += np.flipud(np.fliplr(w[1,0,:,:]))
        y1[1,0,1:-1,1:-1] += np.flipud(np.fliplr(3*w[0,0,:,:]))
        y1[1,1,1:-1,1:-1] += np.flipud(np.fliplr(3*w[1,0,:,:]))
        
        # reference backward path
        y1 = np.reshape(y1, (1,1,2,50))
        dL = 2.0*(y1 - y0)
        dL1 = np.reshape(dL, (2,2,5,5))
        dL2 = np.zeros((2,2,7,7))
        dL2[0,0,:,:] = half(dL1[0,0,:,:])
        dL2[0,1,:,:] = half(dL1[0,1,:,:])
        dL2[1,0,:,:] = half(dL1[1,0,:,:])
        dL2[1,1,:,:] = half(dL1[1,1,:,:])
        
        # check update
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y1 = Variable(torch.FloatTensor(y0), requires_grad=False)
        y2 = cnet(x2)
        lo = loss(y2, y1)
        lo.backward() # torch gradient
        print('Analytically computed gradient of w:')
        dW0 = cnet.conv0.weight.grad.data.numpy()
        print(dW0)
        
        # reference gradient of w
        x1 = np.reshape(x0, (2,1,5,5))
        dW1 = correlate2d(x1[0,0,:,:], dL1[0,0,1:-1,1:-1], 'valid')
        dW2 = correlate2d(x1[0,0,:,:], dL1[0,1,1:-1,1:-1], 'valid')
        dW1 += correlate2d(x1[1,0,:,:], dL1[1,0,1:-1,1:-1], 'valid')
        dW2 += correlate2d(x1[1,0,:,:], dL1[1,1,1:-1,1:-1], 'valid')
        print('True weight gradient:')
        print(dW1)
        print(dW2)
        assert np.allclose(dW0[0,0,:,:], dW1, 0, 1e-5)
        assert np.allclose(dW0[1,0,:,:], dW2, 0, 1e-5)
        
        print('Analytically computed gradient of b:')
        db0 = cnet.conv0.bias.grad.data.numpy()
        print(db0.transpose())
        
        # reference gradient of b
        db1 = np.array([dL1[0,0,:,:].sum(), dL1[0,1,:,:].sum()])
        db1 += np.array([dL1[1,0,:,:].sum(), dL1[1,1,:,:].sum()])
        print('True bias gradient:')
        print(db1)
        assert np.allclose(db0.transpose(), db1, 0, 1e-5)
    
    
    # test convolution with strides
    def test_conv_with_strides(self):    
        print('\n')
        
        # create net
        cnet = Sequential(
            ('unfla', Unflatten([25], [1,5,5])),
            ('conv0', Conv([1,5,5], [1,3,3], stride=2, padding=1))
        )
        print(cnet)
        
        # weights
        w = cnet.conv0.w
        b = cnet.conv0.b
        print('Weight tensor:')
        print(w)
        print('bias:')
        print(b)
        
        # test input
        x0 = np.random.randn(1, 25).astype('float32')
        y0 = np.random.randn(1, 9).astype('float32')
        print('Input:')
        x1 = np.reshape(x0, (5,5))
        print(x1)
        
        # predict and check cross-correlation
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x2)
        y2 = y2.data.numpy()
        print('Predictions Conv for delta input:')
        print(y2)
        assert y2.shape[2] == 3 and y2.shape[3] == 3
        print('Reference predictions (central column):')
        y1 = np.sum(x1[:2,1:4]*w[0,0,1:,:]) + b
        print(y1)
        assert np.allclose(y1[0], y2[0,0,0,1], 0, 1e-5)
        y1 = np.sum(x1[1:4,1:4]*w) + b
        print(y1)
        assert np.allclose(y1[0], y2[0,0,1,1], 0, 1e-5)
        y1 = np.sum(x1[3:,1:4]*w[0,0,:2,:]) + b
        print(y1)
        assert np.allclose(y1[0], y2[0,0,2,1], 0, 1e-5)
    
    
    # test convolution with strides gradient
    def test_conv_with_strides_gradient(self):    
        print('\n')
        
        # create net
        eta = 0.1
        cnet = Sequential(
            ('unfla', Unflatten([25], [1,5,5])),
            ('conv0', Conv([1,5,5], [1,3,3], stride=2, padding=1)),
            ('flatt', Flatten([1,3,3]))
        )
        print(cnet)
        loss = nn.MSELoss(reduction='sum')
        opt = torch.optim.SGD(cnet.parameters(), lr=eta)
        
        # weights
        w = cnet.conv0.w
        b = cnet.conv0.b
        print('Weight tensor:')
        print(w)
        print('bias:')
        print(b)
        
        # test input
        x0 = np.random.randn(1, 25).astype('float32')
        y0 = np.random.randn(1, 9).astype('float32')
        x1 = np.reshape(x0, (5,5))
        
        # predict 
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y1 = Variable(torch.FloatTensor(y0), requires_grad=False)
        y2 = cnet(x2)
        y = y2.data.numpy()
        
        # reference backward path
        y3 = np.reshape(y, (1,9))
        dL = 2.0*(y3 - y0)
        
        # check update
        lo = loss(y2, y1)
        lo.backward() # torch gradient
        print('Analytically computed gradient of w:')
        dW0 = cnet.conv0.weight.grad.data.numpy()
        print(dW0)
        dL1 = np.reshape(dL, (3,3))
        print('True weight gradient:')
        y1 = np.sum(x1[::2,::2]*dL1)
        print(y1)
        assert np.allclose(y1, dW0[0,0,1,1], 0, 1e-5)
    
    
    # test SAME padding MAXPOOLING
    def test_max_pooling_same(self):
        print('\n')    
        batch = 1
        cnet = Sequential(
            ('unfla', Unflatten([35], [1,5,7])),
            ('pool0', MaxPool([1,5,7], kernel_size=(3,3), stride=(2,3), padding=(1,1)))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,35).astype('float32')
        
        # compute reference forward path
        x1 = np.reshape(x0, (batch,1,5,7))
        print('Input:')
        print(x1)
        y1 = np.zeros((3,3), dtype=np.float32)
        y1[0,0] = x1[0,0,:2,:2].max()
        y1[0,1] = x1[0,0,:2,2:5].max()
        y1[0,2] = x1[0,0,:2,5:].max()
        y1[1,0] = x1[0,0,1:4,:2].max()
        y1[1,1] = x1[0,0,1:4,2:5].max()
        y1[1,2] = x1[0,0,1:4,5:].max()
        y1[2,0] = x1[0,0,3:,:2].max()
        y1[2,1] = x1[0,0,3:,2:5].max()
        y1[2,2] = x1[0,0,3:,5:].max()
        
        # predict and check
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x2)
        y2 = y2.data.numpy()
        print('Predictions max pooling:')
        print(y2)
        print('Reference predictions:')
        print(y1)
        assert np.allclose(y2, y1, 0, 1e-5)
    
    
    # test SAME padding AVERAGE
    def test_avg_pooling_same(self):
        print('\n')    
        batch = 1    
        cnet = Sequential(
            ('unfla', Unflatten([35], [1,5,7])),
            ('pool0', AvgPool([1,5,7], kernel_size=(3,5), stride=(2,3), padding=(1,2)))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,35).astype('float32')
        
        # predict
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x2)
        y2 = y2.data.numpy()
        print('Predictions average pooling:')
        print(y2)
        
        print('Reference predictions:')
        x2 = np.reshape(x0, (5,7))
        x1 = np.zeros((7,11))
        x1[1:6,2:9] = x2
        y1 = np.zeros((3, 3), dtype=np.float32)
        y1[0,0] = x1[:3,:5].mean()
        y1[0,1] = x1[:3,3:8].mean()
        y1[0,2] = x1[:3,6:].mean()
        y1[1,0] = x1[2:5,:5].mean()
        y1[1,1] = x1[2:5,3:8].mean()
        y1[1,2] = x1[2:5,6:].mean()
        y1[2,0] = x1[4:,:5].mean()
        y1[2,1] = x1[4:,3:8].mean()
        y1[2,2] = x1[4:,6:].mean()
        print(y1)
        assert np.allclose(y2, y1, 0, 1e-5)
    
    
    # test local response normalization
    def test_lrn(self):
        print('\n')
        batch = 1
        cnet = Sequential(
            ('unfla', Unflatten([16], [4,2,2])),
            ('lrn00', LRN([4,2,2], 3, 2.0, 0.0001, 0.75))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,16).astype('float32')
        
        # compute reference forward path
        x1 = np.reshape(x0, (batch,4,2,2))
        y1 = np.zeros(x1.shape, dtype=np.float32)
        sqr_sum = x1[0,0,:,:]**2
        for j in range(4):
            if j < 3:
                sqr_sum += x1[0,j+1,:,:]**2
            if j > 1:
                sqr_sum -= x1[0,j-2,:,:]**2
            denom = (2.0 + 0.0001*sqr_sum)**(-0.75)
            y1[0,j,:,:] = x1[0,j,:,:]*denom
        
        # predict and check
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x2)
        y2 = y2.data.numpy()
        print('Predictions local response normalization:')
        print(y2)
        print('Reference predictions:')
        print(y1)
        assert np.allclose(y2, y1, 0, 1e-2)
    
    
    # test crop
    def test_crop(self):
        print('\n')
        batch = 2
        cnet = Sequential(
            ('unfla', Unflatten([32], [2,4,4])),
            ('crop0', Crop([2,4,4], 2, 2))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,32).astype('float32')
        x1 = np.reshape(x0, (batch,2,4,4))
        print('Input:')
        print(x1)
        
        # crop in evaluation mode
        cnet.eval()
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y2 = cnet(x2)
        y2 = y2.data.numpy()
        print('Output crop evaluation mode:')
        print(y2)
        y1 = x1[:,:,1:3,1:3]
        assert np.allclose(y2, y1, 0, 1e-5)
        
        # crop in training mode
        cnet.train()
        y2 = cnet(x2)
        y2 = y2.data.numpy()
        print('Output crop training mode:')
        print(y2)
    
    
    # test whitening
    def test_whitening(self):
        print('\n')
        batch = 1
        cnet = Sequential(
            ('unfla', Unflatten([32], [2,4,4])),
            ('crop0', Crop([2,4,4], 2, 2)),
            ('white', Whiten([2,2,2]))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,32).astype('float32')
        x1 = np.reshape(x0, (batch,2,4,4))
        print('Input:')
        print(x1)
        
        # reference whitening
        y1 = x1[:,:,1:3,1:3] # central crop
        y2 = np.zeros((batch,2,2,2), dtype=np.float32)
        for i in range(2):
            stddev = y1[0,i,:,:].std(ddof=1) # numpy uses biased estimator if ddof is not set to 1
            if stddev < 1.0/2.0:
                stddev = 0.5
            y2[0,i,:,:] = (y1[0,i,:,:] - y1[0,i,:,:].mean())/stddev
        
        # predict whiten
        cnet.eval()
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y3 = cnet(x2)
        y3 = y3.data.numpy()
        print('Output whitening:')
        print(y3)
        print('Reference:')
        print(y2)
        assert np.allclose(y3, y2, 0, 1e-5)
    
    
    # test flipLR
    def test_flipLR(self):
        print('\n')
        batch = 2
        cnet = Sequential(
            ('unfla', Unflatten([32], [2,4,4])),
            ('flip0', RandomFlipLR([2,4,4]))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,32).astype('float32')
        x1 = np.reshape(x0, (batch,2,4,4))
        print('Input:')
        print(x1)
        
        # predict flip
        cnet.train()
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y3 = cnet(x2)
        y3 = y3.data.numpy()
        print('Output flipLR:')
        print(y3)
        y3 = cnet(x2)
        y3 = y3.data.numpy()
        print('Output flipLR:') # 2 times to see at least one flip
        print(y3)
    
    
    # test random brightness
    def test_random_brightness(self):
        print('\n')
        batch = 2
        cnet = Sequential(
            ('unfla', Unflatten([32], [2,4,4])),
            ('rndbr', RandomBrightness([2,4,4], 1.0))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,32).astype('float32')
        x1 = np.reshape(x0, (batch,2,4,4))
        print('Input:')
        print(x1)
        
        # predict rb
        cnet.train()
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y3 = cnet(x2)
        y3 = y3.data.numpy()
        print('Output RandomBrightness:')
        print(y3)
        print(y3 - x1)
    
    
    # test random contrast
    def test_random_contrast(self):
        print('\n')
        batch = 2
        cnet = Sequential(
            ('unfla', Unflatten([32], [2,4,4])),
            ('rndco', RandomContrast([2,4,4], 0.1, 10.0))
        )
        print(cnet)
        
        # test dataset
        x0 = np.random.randn(batch,32).astype('float32')
        x1 = np.reshape(x0, (batch,2,4,4))
        x1[0,0,:,:] = x1[0,0,:,:] - x1[0,0,:,:].mean()
        x1[0,1,:,:] = x1[0,1,:,:] - x1[0,1,:,:].mean()
        x1[1,0,:,:] = x1[1,0,:,:] - x1[1,0,:,:].mean()
        x1[1,1,:,:] = x1[1,1,:,:] - x1[1,1,:,:].mean()
        print('Input:')
        print(x1)
        
        # predict rc
        cnet.train()
        x2 = Variable(torch.FloatTensor(x0), requires_grad=False)
        y3 = cnet(x2)
        y3 = y3.data.numpy()
        print('Output RandomContrast:')
        print(y3)
        print(y3/x1)
    
    
    def test_Trainer(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.uniform(0, 10, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.uniform(0, 10, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        
        # CREATE A TRAINER
        trs = Trainingstate()
        lg = Logger(loglevel=logging.ERROR)
        my_trainer = SupervisedTrainer(lg, model, criterion, optimizer, trainingstate=trs, 
            model_filename="testcase",  model_keep_epochs=True)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=4,
                                        validation_set=dataset_valid)
        # ASSERT EPOCH
        assert trs.state["best_validation_loss"][0] == 4
        # ASSERT LOSS
        assert 4.380 > trs.state["best_validation_loss"][1]
        loss_epoch_4 = trs.state["best_validation_loss"][1]
        
        # RESTORE STATE
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion,
            optimizer, trainingstate=trs, model_filename="testcase", precision=np.float32)
        
        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        validation_set=dataset_valid, 
                                        epochs=1)
        # ASSERT EPOCH
        assert trs.state["training_loss[]"][-1][0] == 5
       
        # ASSERT LOSS
        assert loss_epoch_4 > trs.state["best_validation_loss"][1]
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))
    
    
    def test_trainer_cuda(self):
        np.random.seed(17)
        torch.manual_seed(17)
        
        # test makes only sense when your machine has cuda enabled
        if not torch.cuda.is_available():
            print('\nWarning: cannot run this test - Cuda is not enabled on your machine.')
            return
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.uniform(0, 10, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.uniform(0, 10, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        
        # CREATE A TRAINER
        trs = Trainingstate()
        lg = Logger(loglevel=logging.ERROR)
        my_trainer = SupervisedTrainer(lg, model, criterion, optimizer, trainingstate=trs, 
            model_filename="testcase",  model_keep_epochs=True, use_cuda = True)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=4,
                                        validation_set=dataset_valid)
        # ASSERT EPOCH
        assert trs.state["best_validation_loss"][0] == 4
        # ASSERT LOSS
        assert 4.380 > trs.state["best_validation_loss"][1]
        loss_epoch_4 = trs.state["best_validation_loss"][1]
        
        # RESTORE STATE
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion,
            optimizer, trainingstate=trs, model_filename="testcase", precision=np.float32)
        
        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        validation_set=dataset_valid, 
                                        epochs=1)
        # ASSERT EPOCH
        assert trs.state["training_loss[]"][-1][0] == 5
       
        # ASSERT LOSS
        assert loss_epoch_4 > trs.state["best_validation_loss"][1]
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))
                
    
    def test_trainingstate_update(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(10, 256)
                self.fc2 = nn.Linear(256, 84)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.relu(self.fc2(x))
                    x = F.softmax(x, dim=1)
                    return x
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(Net().parameters(), lr=0.01)
        
        ts = Trainingstate()
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = 0, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     args = { "args" : 1 , "argv" : 2})
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = 0, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     args = { "args" : 1 , "argv" : 2})
        
        
    def test_trainingstate_persistency(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(10, 256)
                
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.softmax(x, dim=1)
                    return x
        
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(Net().parameters(), lr=0.01)
        
        ts = Trainingstate()
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = 0, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     validation_dataset = dataset_valid,
                     args = { "args" : 1 , "argv" : 2})
        
        ts2 = Trainingstate()
        ts.save_state("test.pth.tar")
        ts2.load_state("test")

        ts.save_state("test")
        ts2.load_state("test.pth.tar")

        ts.save_state("test.pth.tar")
        ts2.load_state("test.pth.tar")

        assert ts2["validation_accuracy[]"][-1][0] == ts["validation_accuracy[]"][-1][0]
        assert ts2["validation_loss[]"][-1][0] == ts["validation_loss[]"][-1][0]
        assert ts2["lrate[]"][-1][1] == ts["lrate[]"][-1][1]
        assert ts2["model_trainable_params"] == ts["model_trainable_params"]
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(ts.extension):
                os.remove(os.path.join(dir,file))
                
    def test_transform_model_from_cpu_to_cuda(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # test makes only sense when your machine has cuda enabled
        if not torch.cuda.is_available():
            print('\nWarning: cannot run this test - Cuda is not enabled on your machine.')
            return
        
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=False, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trs = Trainingstate()
        
        # CREATE A CPU TRAINER
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase_cpu",  model_keep_epochs=True, use_cuda = False)
        
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=3,
                                        validation_set=dataset_valid)
        
        loss_epoch_3_cpu = trs["validation_loss[]"][-1][1]
        
        # CREATE A CUDA TRAINER
        state_cpu = Trainingstate("testcase_cpu_epoch_2")
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=state_cpu, model_filename="testcase_cpu",  model_keep_epochs=True, use_cuda = True)
        
        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=1,
                                        validation_set=dataset_valid)       

        loss_epoch_3_cuda = state_cpu["validation_loss[]"][-1][1]
        
        # ASSERT             
        assert np.allclose(loss_epoch_3_cuda, loss_epoch_3_cpu, 1e-5) and np.allclose(0.5029625296592712, loss_epoch_3_cpu, 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))


    def test_transform_model_from_cuda_to_cpu(self):
        np.random.seed(17)
        torch.manual_seed(17)
       
        # test makes only sense when your machine has cuda enabled
        if not torch.cuda.is_available():
            print('\nWarning: cannot run this test - Cuda is not enabled on your machine.')
            return
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=False, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trs = Trainingstate()
        
         # CREATE A CUDA TRAINER
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase_cuda",  model_keep_epochs=True, use_cuda = True)
        
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=3,
                                        validation_set=dataset_valid)
        
        loss_epoch_3_cuda = trs["validation_loss[]"][-1][1]
        
        state_cuda = Trainingstate("testcase_cuda_epoch_2")
        
        # CREATE A CUDA TRAINER
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=state_cuda, model_filename="testcase_cpu",  model_keep_epochs=True, use_cuda = False)
        
        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=1,
                                        validation_set=dataset_valid)       

        loss_epoch_3_cpu = state_cuda["validation_loss[]"][-1][1]
        
        # ASSERT             
        assert np.allclose(loss_epoch_3_cuda, loss_epoch_3_cpu, 1e-5) and np.allclose(0.5029625296592712, loss_epoch_3_cpu, 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))
                
                
    def test_transform_model_from_double_to_float(self):
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
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.uniform(0, 1, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.uniform(0, 1, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        trs = Trainingstate()
        
        # CREATE A FLOAT TRAINER
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=100, shuffle=True, sampler=None, batch_sampler=None)
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase_float", 
            model_keep_epochs=True, use_cuda = False, precision = np.float32)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=5,
                                        validation_set=dataset_valid)
        
        single_precision_loss = trs.state["best_validation_loss"][1]
        assert np.allclose(single_precision_loss, SupervisedAnalyzer.evaluate(model, 
            criterion, dataset_valid, batch_size = 100)["loss"], 1e-5)
        
        # CREATE A DOUBLE DATASET
        dataset = TensorDataset(x.double(), y.double())
        dataset_valid = TensorDataset(x_valid.double(), y_valid.double())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
      
        state = Trainingstate("testcase_float_epoch_5")
        model = Trainingstate.transform_model(model, optimizer, np.float64)
     
        # ASSERT INFERENCE
        assert np.allclose(single_precision_loss, SupervisedAnalyzer.evaluate(model, 
            criterion, dataset_valid, batch_size = 100)["loss"], 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))

    def test_reset_best_validation_model(self):
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
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.uniform(0, 1, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.uniform(0, 1, 100).reshape(100,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        trs = Trainingstate()
        
        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase_valid",  model_keep_epochs=True)
        
        # START TRAINING
        # Validation loss
        # ..  (9, 0.9189968437328946, 10000), **
        #    (10, 0.9509468257427217, 10000), 
        #    (11, 0.8643921442590652, 10000)]
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=10,
                                        validation_set=dataset_valid)
        # Validation Error
        best_val_loss = trs.state["best_validation_loss"][1]
        
        
        # RESET STATE
        model = Net()
        trs.load_weights_best_validation_(model, optimizer)
        
        loss_after = SupervisedAnalyzer.evaluate(model, criterion, dataset_valid, batch_size=10)["loss"]
        assert np.allclose(loss_after, best_val_loss , 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))


    def test_reset_best_training_model(self):
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
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=100, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        trs = Trainingstate()
        
        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase",  model_keep_epochs=True)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=10,
                                        validation_set=dataset_valid)

        #[(1, 14.068070793151849, 100), (2, 4.549037075042724, 100), (3, 3.409626841545108, 100), (4, 2.536898219585421, 100), (5, 2.0484979152679434, 100), (6, 2.3173243045806857, 100), (7, 2.3507825613021844, 100), (8, 1.442621475458147, 100), (9, 1.3470908164978037, 100), (10, 1.7832313299179066, 100)])])
        # Training Error
        #assert 4 > trs.state["best_training_loss"][1]
        loss_before = SupervisedAnalyzer.evaluate(model, criterion, dataset, batch_size = 100)["loss"]
       
        # RESET STATE
        model = Net()
        trs.load_weights_best_training_(model, optimizer)
        
        loss_after = SupervisedAnalyzer.evaluate(model, criterion, dataset, batch_size = 100)["loss"]
        assert np.allclose(loss_before, loss_after, 1e1)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))


    def test_update_state_without_validation_loss(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(10, 256)
                
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = F.softmax(x, dim=1)
                    return x
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(Net().parameters(), lr=0.001)
        
        ts = Trainingstate()
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = 0, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     validation_dataset = dataset_valid,
                     args = { "args" : 1 , "argv" : 2})

        assert len(ts["validation_accuracy[]"]) == 1
        
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = None, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = None, 
                     args = { "args" : 1 , "argv" : 2})
        assert len(ts["validation_accuracy[]"]) == 1
        
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = 0, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     validation_dataset = dataset_valid,
                     args = { "args" : 1 , "argv" : 2})

        assert len(ts["validation_accuracy[]"]) == 2

        ts = Trainingstate()
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = None, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     args = { "args" : 1 , "argv" : 2})
        
        assert len(ts["validation_accuracy[]"]) == 0
    
    def test_analyzer(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return x
    
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        
        model = Net()
        criterion = nn.MSELoss()
        model = Trainingstate.transform_model(model, None, precision=np.float32)
        self.assertTrue(SupervisedAnalyzer.evaluate(model, criterion, dataset_valid,  batch_size=1)["loss"] < 1.)
        self.assertTrue(np.allclose(SupervisedAnalyzer.evaluate(model, criterion, dataset_valid,  batch_size= 1)["loss"],
                                    SupervisedAnalyzer.evaluate(model, criterion, dataset_valid,  batch_size=10)["loss"]))
       
    def test_predictor(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                return x
    
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        dataset_valid = UnsupTensorDataset(x_valid.float())
        
        model = Net()
        model = Trainingstate.transform_model(model, None, precision=np.float32)
        self.assertTrue(type(Predictor.predict(model, dataset_valid)) == torch.Tensor)
        
        # SIMPLIFIED Interface
        y0 = Predictor.predict(model, x_valid.float().numpy(), batch_size = 100, output_transform=torch.sigmoid)
        self.assertTrue(type(y0) == np.ndarray)
   
        # SIMPLIFIED Interface with Tensors
        y1 = Predictor.predict(model, x_valid.float(), batch_size = 100, output_transform=torch.sigmoid)
        self.assertTrue(type(y1) == torch.Tensor)
        
        assert np.allclose(y0, y1.numpy())
   
        
        
    def test_hooks(self):
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, 
            sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        
        backward_called = False
        eval_called = False
        
        def eval_hook(ctx, output, targets, loss):
            assert isinstance(loss, torch.Tensor)
            import intentionally_wrong_import
        
        my_trainer = SupervisedTrainer(Logger(), model, criterion, 
            optimizer, model_filename="testcase",  model_keep_epochs=True,
                                        after_eval_hook=eval_hook)
        try:
            my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=1,
                                        validation_set=dataset_valid)
        except ModuleNotFoundError:
            eval_called = True
        
        def backward(output, targets, loss):
            assert isinstance(loss, torch.Tensor)
            import intentionally_wrong_import
            
        my_trainer = SupervisedTrainer(Logger(), model, criterion, 
            optimizer, model_filename="testcase",  model_keep_epochs=True,
                                        after_backward_hook=backward)
        try:
            my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=1,
                                        validation_set=dataset_valid)
        except ModuleNotFoundError:
            backward_called = True
        
        
        assert backward_called == True == eval_called
        
        
    def test_classification(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        y = 2 * torch.from_numpy(np.random.binomial(1, 0.5, 10000).reshape(10000,1)) - 1
        
        x_valid = torch.from_numpy(np.random.normal(0, 1, 100).reshape(100,1))

        # -1/1 Encoding
        y_valid = 2 * torch.from_numpy(np.random.binomial(1, 0.5, 100).reshape(100,1)) - 1

        # 0/1 Encoding
        y_valid2 = torch.from_numpy(np.random.binomial(1, 0.5, 100).reshape(100,1))

        dataset = TensorDataset(x.double(), y.double())
        dataset_valid = TensorDataset(x_valid.double(), y_valid.double())
        dataset_valid2 = TensorDataset(x_valid.double(), y_valid2.double())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        loss = nn.BCEWithLogitsLoss(reduction='sum')
        trs = Trainingstate()
        
        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        my_trainer = ClassificationTrainer(Logger(), model, loss, 
            optimizer, trainingstate=trs, model_filename="testcase",  model_keep_epochs=False, precision=np.float64)
        my_trainer.fit(dataloader_training=dataloader_trainingdata, epochs=2, validation_set=dataset_valid)
        output = Predictor.predict(model, x_valid.numpy())
        predicts = Predictor.classify(output)
        accuracy = Predictor.compute_accuracy(predicts, y_valid)
        assert accuracy > 0.48

        predicts = Predictor.classify(output)
        accuracy = Predictor.compute_accuracy(predicts, y_valid2)
        assert accuracy > 0.48
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))

    
    def test_unsupervised(self):
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
                x = self.fc1(x)
                x = self.fc2(x)
                return x
    
        
        x_valid = torch.from_numpy(np.random.normal(0, 0.1, 1000).reshape(1000,1)).float()
        dataset_valid = UnsupTensorDataset(x_valid)
        x = torch.from_numpy(np.random.normal(0, 0.1, 1000).reshape(1000,1)).float()
        dataset = UnsupTensorDataset(x)
        dataloader_training = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        trs = Trainingstate()

        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01/10)
        my_trainer = UnsupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase",  model_keep_epochs=True)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_training,
                                        epochs=3,
                                        validation_set=dataset_valid)
        
        assert trs["training_loss[]"][-1][1] < trs["training_loss[]"][0][1]
        
        
        # TEST SIMPLIFIED INTERFACE 
        model = Net()
        criterion = nn.MSELoss(size_average=False)
        trs = Trainingstate()
        
        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01/10)
        my_trainer = UnsupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase",  model_keep_epochs=True)
       
         # START TRAINING
        my_trainer.fit(dataloader_training=(x.numpy(), 10),
                                        epochs=3,
                                        validation_set=x_valid.numpy())
        
        assert len(trs["training_loss[]"]) == 3
        assert trs["training_loss[]"][-1][1] < trs["training_loss[]"][0][1]
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))

    def test_siamese(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(1, 10)
                self.fc3 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x_l, x_r = x
                x_l, x_r = self.fc1(x_l), self.fc2(x_r)
                return  F.pairwise_distance(x_l, x_r)
    
        
        x_valid = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.random.binomial(1,0.5,10000)) 
        dataset_valid = SiameseTensorDataset(x_valid.float(), x_valid.float(), y_valid.float(), y_valid.float())
        
        x = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        y = torch.from_numpy(np.random.binomial(1,0.5,10000)) 
        dataset = SiameseTensorDataset(x.float(), x.float(), y.float(), y.float())
        dataloader_training = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = Contrastive(size_average=False)
        trs = Trainingstate()

        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        my_trainer = SiameseTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase",  model_keep_epochs=False)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_training,
                                        epochs=2,
                                        validation_set=dataset_valid)
        
        assert trs["best_validation_loss"][1] < 1.
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))
                
                
    def test_lr_scheduler(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.uniform(0, 10, 1000).reshape(1000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.uniform(0, 10, 1000).reshape(1000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=False, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ts = Trainingstate()
        lg = Logger(loglevel=logging.ERROR)
    
        # EARLY STOPPING
        earlystop = StepLR_earlystop(optimizer, ts, model, step_size=10, nsteps=3, logger=lg, 
            patience=1)
  
        # CREATE A TRAINER
        my_trainer = SupervisedTrainer(lg, model, criterion,
            optimizer, trainingstate=ts, scheduler = earlystop, model_filename="testcase",  model_keep_epochs=True)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=30,
                                        validation_set=dataset_valid)
        
        # assert monotony in lrate and validation loss
        assert any([ts["lrate[]"][i-1][1] > ts["lrate[]"][i][1] for i in range(1, len(ts["lrate[]"]))])
        assert any([ts["validation_loss[]"][i-1][1] > ts["validation_loss[]"][i][1] for i in range(1, len(ts["validation_loss[]"]))])
        assert ts["best_validation_loss"][0] < 30
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(ts.extension):
                os.remove(os.path.join(dir,file))
                
                
    def test_convergence_criterion(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 100)
                self.fc2 = nn.Linear(100, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x
        
        x_valid = torch.from_numpy(np.random.normal(0, 1, 100).reshape(100,1))
        dataset_valid = UnsupTensorDataset(x_valid.float())
        x = torch.from_numpy(np.random.normal(0, 1, 1000).reshape(1000,1))
        dataset = UnsupTensorDataset(x.float())
        dataloader_training = DataLoader(dataset, batch_size=10, shuffle=False, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        trs = Trainingstate()

        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01/10)
        my_trainer = UnsupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase", convergence_eps = 1e-3, model_keep_epochs=True)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_training,
                                        epochs=25,
                                        validation_set=dataset_valid)
        
        assert trs["training_loss[]"][-1][0] < 25
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))
                
    def test_combined_retraining(self):
        np.random.seed(17)
        torch.manual_seed(17)
        #
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x
    
        
        x_valid = torch.from_numpy(np.random.normal(0, 0.1, 10).reshape(10,1))
        dataset_valid = UnsupTensorDataset(x_valid.float())
        
        x = torch.from_numpy(np.random.normal(0, 0.1, 1000).reshape(1000,1))
        dataset = UnsupTensorDataset(x.float())
        dataloader_training = DataLoader(dataset, batch_size=5, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        trs = Trainingstate()

        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)
        my_trainer = UnsupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase", combined_training_epochs = 4)
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_training,
                                        epochs=2,
                                        validation_set=dataset_valid)
        assert len(trs["training_loss[]"]) == 6
        
        # TEST PERSISTED MODEL
        retrained_state = Trainingstate(str("testcase" + trs.combined_retraining_pattern + "_epoch_4"))
        
        assert retrained_state["training_loss[]"][-1][1] < retrained_state["training_loss[]"][-4][1]    

        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))
                

    def test_resume_persisted_training(self):
        np.random.seed(17)
        torch.manual_seed(17)
        
        # DEFINE a neural network
        class Net(nn.Module):
        
            def __init__(self):
                super(Net, self).__init__()
                self.fc1 = nn.Linear(1, 10)
                self.fc2 = nn.Linear(10, 1)
                
                # Initialization
                def weights_init_normal(m):
                    if type(m) == nn.Linear:
                        nn.init.normal_(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = torch.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=False, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss(reduction='sum')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trs = Trainingstate()
        
        # CREATE A CPU TRAINER
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=trs, model_filename="testcase_cpu",  model_keep_epochs=True, use_cuda = False)
        
        
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=3,
                                        validation_set=dataset_valid)
        
        loss_epoch_3_cpu = trs["validation_loss[]"][-1][1]
        
        # CREATE A NEW CPU TRAINER
        state_cpu = Trainingstate("testcase_cpu_epoch_2")
        my_trainer = SupervisedTrainer(Logger(loglevel=logging.ERROR), model, criterion, 
            optimizer, trainingstate=state_cpu, model_filename="testcase_cpu",  model_keep_epochs=True, use_cuda = False)
        
        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=1,
                                        validation_set=dataset_valid)       

        loss_epoch_3_cpu2 = state_cpu["validation_loss[]"][-1][1]
        
        # ASSERT
        assert np.allclose(loss_epoch_3_cpu2, loss_epoch_3_cpu, 1e-5) and np.allclose(5.029627222061157, loss_epoch_3_cpu, 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trs.extension):
                os.remove(os.path.join(dir,file))
 

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('test', default="", metavar="",
                        help="Execute a specific test")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]
    if argv.test is not "":
        eval(str("TestUmmon()." + argv.test + '()'))
    else:
        unittest.main()
