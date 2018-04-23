#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import unittest
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
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
            ('line0', Linear([5], 7, 'xavier_uniform', 0.001)),
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
        loss = nn.MSELoss()
        mse = loss(x1, y1).data.numpy()
        print('MSE:     ', mse[0])
        mse_true = ((x0 - y0)**2).mean() # pyTorch divides by n_dim instead of 2
        print('True MSE:', mse_true)
        assert np.allclose(mse, mse_true, 0, 1e-3)
        
        # test log likelihood loss function
        y1 = Variable(torch.LongTensor(2*y0[:,2]), requires_grad=False) # pytorch takes class index, not one-hot coding
        loss = nn.NLLLoss()
        LL = loss(x1, y1).data.numpy()
        print('LL:     ', LL[0])
        # should be LL_true = (-y0*np.nan_to_num(np.log(x0))).sum(axis=1).mean(), but pytorch expects x1 to be already log'd by Log Softmax 
        LL_true = (-y0*x0).sum(axis=1).mean()
        print('True LL:', LL_true)
        assert np.allclose(LL, LL_true, 0, 1e-3)
        
        # test pytorch cross entropy loss function (=logsoftmax + NLL)
        loss = nn.CrossEntropyLoss()
        ce = loss(x1, y1).data.numpy()
        print('CE:      ', ce[0])
        # pytorch CE is combination of log softmax and log likelihood
        ce_true = (-x0[:,2] + np.log(np.exp(x0).sum(axis=1))).mean()
        print('True CE: ', ce_true)
        assert np.allclose(ce, ce_true, 0, 1e-3)
        
        # test binary cross entropy
        loss = nn.BCELoss()
        y1 = Variable(torch.FloatTensor(y0), requires_grad=False)
        bce = loss(x1, y1).data.numpy()
        print('BCE:     ', bce[0])
        bce_true = ((np.nan_to_num(-y0*np.log(x0)-(1-y0)*np.log(1-x0))).mean(axis=1)).mean()
        print('True BCE:', bce_true) # pytorch takes mean across dimensions instead of sum
        assert np.allclose(bce, bce_true, 0, 1e-3)
        
        # test pytorch combined sigmoid and bce
        loss = nn.BCEWithLogitsLoss()
        bce = loss(x1, y1).data.numpy()
        print('BCEL:    ', bce[0])
        bce_true = ((np.nan_to_num(-y0*np.log(sigmoid(x0))-(1-y0)*np.log(1-sigmoid(x0)))).mean(axis=1)).mean()
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
        print('HingeE:  ', hinge[0])
        hinge_true = (x0[:3].sum() + np.maximum(0, 1 - x0[3:]).sum())/6.0
        print('TrueHinE:', hinge_true)
        assert np.allclose(hinge, hinge_true, 0, 1e-3)
        
        # test true standard hinge loss
        loss = nn.MarginRankingLoss(margin=1.0)
        dummy = torch.FloatTensor(6,1).zero_()
        # dummy variable must have same size as x1, but must be 0
        hinge = loss(x1, dummy, y1).data.numpy()
        print('Hinge:   ', hinge[0])
        hinge_true = ((np.maximum(0, 1 - x0*y0)).sum(axis=1)).mean()
        print('True Hin:', hinge_true)
        assert np.allclose(hinge, hinge_true, 0, 1e-3)
        
    #     # test 3 pixel error metric function
    #         loss = Loss('3Pix', [7], cnet, [], '3_pixel')
    #         cnet.init_network()
    # 
    #         x1 = np.zeros((batch+1, 7), dtype=np.float32)
    #         y1 = np.zeros((batch+1, 7), dtype=np.float32)
    #         # simulate softmax
    #         x1[:, 0] = -1
    #         x1[:, 1] = -2
    #         x1[:, 2] = -3
    #         x1[:, 3] = -4
    #         x1[:, 4] = -5
    #         x1[:, 5] = -6
    #         x1[:, 6] = -7
    # 
    #         y1[0][3] = 1
    #         y1[1][3] = 1
    #         y1[2][3] = 1
    #         y1[3][3] = 1
    #         y1[4][3] = 1
    #         y1[5][3] = 1
    #         y1[6][3] = 1
    # 
    #         pix = cnet.avg_loss(x1, y1)
    #         print('3 Pix:   ', pix)
    # 
    #         pix_true = 0
    #         x1_gt = np.array([0.05, 0.2, 0.5, 0.2, 0.05])
    #         for i in range(0, x1.shape[0]):
    #             mul = x1_gt*x1[i,1:6]
    #             pix_true = pix_true - mul.sum()
    # 
    # 
    #         print('True 3 Pix:', pix_true)
    #         assert np.allclose(pix, pix_true, 0, 1e-3)
    
    
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
                lo =  ((y - y2)**2).mean()
            elif lossfunc == 'cross_entropy': # binary cross entropy
                lo = ((np.nan_to_num(-y*np.log(y2)-(1-y)*np.log(1-y2))).mean(axis=1)).mean()
            elif lossfunc == 'log_likelihood': # log likelihood
                lo = (-y*y2).sum(axis=1).mean()
            
            return lo
        
        def check_grad_activation(lossfunc, act):
            
            print('\n')
            print("Loss function: {}, Activation: {}".format(lossfunc, act))
            eta = 0.5
            batch = 4
            
            # activation
            if act == 'sigmoid':
                nonl = nn.Sigmoid()
            elif act == 'logsoftmax':
                nonl = nn.LogSoftmax()
            
            # net
            cnet = Sequential(
                ('line0', Linear([3], 5, 'xavier_normal')),
                ('nonl0', nonl)
            )
            print(cnet)
            
            # loss
            if lossfunc == 'mse':
                loss = nn.MSELoss()
            elif lossfunc == 'cross_entropy':
                loss = nn.BCELoss()
            elif lossfunc == 'log_likelihood': 
                loss = nn.NLLLoss()
            
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
        eta = 0.1
        batch = 4
        cnet = Sequential(
            ('line0', Linear([3], 2, 'xavier_normal'))
        )
        loss = nn.BCEWithLogitsLoss()
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
            db = (1/batch)*dL.sum(axis=0).reshape(2,1)
            b = b - eta*db
            dW = (1/batch)*np.dot(dL.transpose(), x0[i*batch:(i+1)*batch,:])
            w = w - eta*dW
        
        print('Ref. weight matrix:')
        print(w)
        print('Ref. bias:')
        print(b.T[0])

        # fit
        with Logger(logdir='', loglevel=20) as lg:
            trn = Trainer(lg, cnet, loss, opt)
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # CREATE A TRAINER
        my_trainer = Trainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase",  model_keep_epochs=True)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=4,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)
        # Validation Error
        # [(1, 0.5116240382194519, 10000), 
        # (2, 0.5512791275978088, 10000), 
        # (3, 0.5019993185997009, 10000), 
        # (4, 0.4970156252384186, 10000), Max
        # (5, 0.5055180191993713, 10000)]
        assert np.allclose(0.4970156252384186,
            trainingsstate.state["best_validation_loss"][1], 1e-5)
        
        # RESTORE STATE
        my_trainer = Trainer(Logger(logdir = '', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase", 
             precision=np.float32)
        
        # RESTART TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        validation_set=dataset_valid, 
                                        epochs=1,
                                        eval_interval=1,
                                        trainingstate=trainingsstate)
        # ASSERT EPOCH
        assert trainingsstate.state["training_loss[]"][-1][0] == 5
        
        # ASSERT LOSS
        assert np.allclose(0.4970156252384186,
            trainingsstate.state["best_validation_loss"][1], 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
    
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

        # CREATE A TRAINER
        my_trainer = Trainer(Logger( logdir = '', log_batch_interval=500), 
                                       model, criterion, optimizer, model_filename="testcase",  
                                       model_keep_epochs=True, use_cuda=True)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=2,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)
        
        assert len(trainingsstate.state["validation_loss[]"]) == 2
        best_validation_loss = trainingsstate.state["best_validation_loss"][1]
        best_training_loss = trainingsstate.state["best_training_loss"][1]
        
        # Validation Error
        # (1, 0.4969218373298645, 10000), Max
        # (2, 0.49693697690963745, 10000), 
        # (3, 0.49688512086868286, 10000), 
        # (4, 0.49684351682662964, 10000), 
        # (5, 0.49691537022590637, 10000)]
        self.assertTrue(np.allclose(0.4969218373298645, best_validation_loss, 1e-5))

        # RESTORE STATE
        my_trainer = Trainer(Logger( logdir = '', log_batch_interval=500), 
                             model, criterion, optimizer, model_filename="testcase", 
                              precision=np.float32, use_cuda=True)

        # RESTART TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=1,
                                        validation_set=dataset_valid, 
                                        eval_interval=2,
                                        trainingstate=trainingsstate)
        # Should improve
        assert len(trainingsstate.state["training_loss[]"]) == 3
        assert np.allclose(trainingsstate.state["best_training_loss"][1], best_training_loss, 1e-2)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # CREATE A CPU TRAINER
        my_trainer = Trainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase_cpu",  model_keep_epochs=True, use_cuda = False)
        
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=3,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)
        
        loss_epoch_3_cpu = trainingsstate["validation_loss[]"][-1][1]
        
        # CREATE A CUDA TRAINER
        my_trainer = Trainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase_cpu",  model_keep_epochs=True, use_cuda = True)
        
        state_cpu = Trainingstate("testcase_cpu_epoch_2")
        
        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=1,
                                        validation_set=dataset_valid, 
                                        eval_interval=2,
                                        trainingstate=state_cpu)       

        loss_epoch_3_cuda = trainingsstate["validation_loss[]"][-1][1]
        
        # ASSERT             
        assert np.allclose(loss_epoch_3_cuda, loss_epoch_3_cpu, 1e-5) and np.allclose(0.5019993185997009, loss_epoch_3_cpu, 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
         # CREATE A CUDA TRAINER
        my_trainer = Trainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase_cuda",  model_keep_epochs=True, use_cuda = True)
        
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=3,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)
        
        loss_epoch_3_cuda = trainingsstate["validation_loss[]"][-1][1]
        
        # CREATE A CUDA TRAINER
        my_trainer = Trainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase_cpu",  model_keep_epochs=True, use_cuda = False)
        
        state_cpu = Trainingstate("testcase_cuda_epoch_2")
        state_cpu = Trainingstate(str("testcase_cuda_epoch_2" + Trainingstate().extension))
       
        
        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=1,
                                        validation_set=dataset_valid, 
                                        eval_interval=2,
                                        trainingstate=state_cpu)       

        loss_epoch_3_cpu = trainingsstate["validation_loss[]"][-1][1]
        
        # ASSERT             
        assert np.allclose(loss_epoch_3_cuda, loss_epoch_3_cpu, 1e-5) and np.allclose(0.5019993185997009, loss_epoch_3_cpu, 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
                os.remove(os.path.join(dir,file))
                
                
    def test_transform_model_from_double_to_float(self):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        model = Net()
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        
        # CREATE A FLOAT TRAINER
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        my_trainer = Trainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase_float", model_keep_epochs=True, use_cuda = False, precision = np.float32)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=5,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)
        
        assert np.allclose(0.4970156252384186,
            trainingsstate.state["best_validation_loss"][1], 1e-5)
        assert np.allclose(0.5055180191993713, Analyzer.evaluate(model, criterion, dataset_valid)["loss"], 1e-5)
        
        # CREATE A DOUBLE DATASET
        dataset = TensorDataset(x.double(), y.double())
        dataset_valid = TensorDataset(x_valid.double(), y_valid.double())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
      
        state = Trainingstate("testcase_float_epoch_5")
        model = uu.transform_model(model, np.float64)
     
        # ASSERT INFERENCE             
        assert np.allclose(0.5055211959813041, Analyzer.evaluate(model, criterion, dataset_valid)["loss"], 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
                os.remove(os.path.join(dir,file))

    def test_reset_best_validation_model(self):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        
        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        my_trainer = Trainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase_valid",  model_keep_epochs=True)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=5,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)
        # Validation Error
        # [(1, 0.5116240382194519, 10000), 
        # (2, 0.5512791275978088, 10000), 
        # (3, 0.5019993185997009, 10000), 
        # (4, 0.4970156252384186, 10000), MAX
        # (5, 0.5055180191993713, 10000)]
        assert np.allclose(0.4970156252384186,
            trainingsstate.state["best_validation_loss"][1], 1e-5)
        
        # ASSERT INFERENCE BEFORE             
        assert np.allclose(0.5055211959813041, Analyzer.evaluate(model, criterion, dataset_valid)["loss"], 1e-5)
        
        # RESET STATE
        model = trainingsstate.load_weights_best_validation(model)
        
        # ASSERT INFERENCE   
        assert np.allclose(0.4970156252384186, Analyzer.evaluate(model, criterion, dataset_valid)["loss"], 1e-5)
        
        # RESET STATE 2
        model = trainingsstate.load_weights_best_validation(model)
        
        # ASSERT INFERENCE   
        assert np.allclose(0.4970156252384186, Analyzer.evaluate(model, criterion, dataset_valid)["loss"], 1e-5)
        
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
                os.remove(os.path.join(dir,file))


    def test_reset_best_training_model(self):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        
        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        my_trainer = Trainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase",  model_keep_epochs=True)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=5,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)

        # Training Error
        # [(1, 0.55760295391082904, 10), 
        # (2, 0.46499215960502555, 10),  MAX
        # (3, 0.56548817157745335, 10), 
        # (4, 0.60142931938171518, 10), 
        # (5, 0.63713452816009752, 10)]
        assert np.allclose(0.46499215960502555,
            trainingsstate.state["best_training_loss"][1], 1e-5)

        # RESET STATE
        model = trainingsstate.load_weights_best_training(model)
        
        # Validation Error
        # [(1, 0.5116240382194519, 10000), 
        # (2, 0.5512791275978088, 10000),  MAX
        # (3, 0.5019993185997009, 10000), 
        # (4, 0.4970156252384186, 10000), 
        # (5, 0.5055180191993713, 10000)]        
        # ASSERT INFERENCE             
        assert np.allclose(0.5512791275978088, Analyzer.evaluate(model, criterion, dataset_valid)["loss"], 1e-5)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
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
        model = uu.transform_model(model, precision=np.float32)
        self.assertTrue(Analyzer.evaluate(model, criterion, dataset_valid,  batch_size=1)["loss"] < 1.)
        self.assertTrue(np.allclose(Analyzer.evaluate(model, criterion, dataset_valid,  batch_size= 1)["loss"],
                                    Analyzer.evaluate(model, criterion, dataset_valid,  batch_size=10)["loss"]))
        self.assertTrue(type(Analyzer.inference(model, dataset_valid, Logger())) == torch.Tensor)
     
        
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.float(), y.float())
        dataset_valid = TensorDataset(x_valid.float(), y_valid.float())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()
        
        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        my_trainer = Trainer(Logger(), model, criterion, 
            optimizer, model_filename="testcase",  model_keep_epochs=True)
        
        def backward(model, output, targets, loss):
            assert not isinstance(output, torch.autograd.Variable)
            assert not isinstance(targets, torch.autograd.Variable)
            assert isinstance(loss, torch.Tensor)
            
        def eval(model, output, targets, loss):
            assert not isinstance(output, torch.autograd.Variable)
            assert not isinstance(targets, torch.autograd.Variable)
            assert isinstance(loss, torch.Tensor)
            
        # START TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        epochs=5,
                                        validation_set=dataset_valid, 
                                        eval_interval=1,
                                        after_backward_hook=backward, 
                                        after_eval_hook=eval)
    def test_classification(self):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = F.sigmoid(self.fc1(x))
                x = self.fc2(x)
                return x
        
        x = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        x_valid = torch.from_numpy(np.random.normal(100, 20, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        
        dataset = TensorDataset(x.double(), y.double())
        dataset_valid = TensorDataset(x_valid.double(), y_valid.double())
        dataloader_trainingdata = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        loss = nn.BCEWithLogitsLoss()
        
        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        my_trainer = Trainer(Logger(), model, loss, 
            optimizer, model_filename="testcase",  model_keep_epochs=False, precision=np.float64)
        
        my_trainer.fit(dataloader_training=dataloader_trainingdata, epochs=1,  validation_set=dataset_valid)
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension):
                os.remove(os.path.join(dir,file))

    
    def test_examples(self):
        import examples.checkstate
        import examples.mnist1
        import examples.validation
        import examples.sine
        examples.sine.example()
        examples.mnist1.example()
        examples.validation.example()
        examples.checkstate.example()
        
        # Clean up
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension) or file.endswith(".log"):
                os.remove(os.path.join(dir,file))
                
    
    def test_unsupervised(self):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x = self.fc1(x)
                x = self.fc2(x)
                return x
    
        
        x_valid = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        dataset_valid = UnsupTensorDataset(x_valid.float())
        x = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        dataset = UnsupTensorDataset(x.float())
        dataloader_training = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()

        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        my_trainer = UnsupervisedTrainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase",  model_keep_epochs=True)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_training,
                                        epochs=5,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)
        
        assert trainingsstate["training_loss[]"][-1][1] < trainingsstate["training_loss[]"][0][1]
        
        xn_valid = np.random.normal(0, 1, 10000).reshape(10000,1).astype(np.float32)
        xn = np.random.normal(0, 1, 10000).reshape(10000,1).astype(np.float32)
        # TEST SIMPLIFIED INTERFACE 
        trainingsstate = my_trainer.fit(dataloader_training=(xn, 10),
                                        epochs=5,
                                        validation_set=xn_valid,
                                        eval_interval=1)
        
        assert trainingsstate["training_loss[]"][-1][1] < trainingsstate["training_loss[]"][0][1]
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
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
                        nn.init.normal(m.weight, mean=0, std=0.1)
                self.apply(weights_init_normal)
        
            def forward(self, x):
                x_l, x_r = x
                x_l, x_r = self.fc1(x_l), self.fc2(x_r)
                return self.fc3(x_l + x_r)
    
        
        x_valid = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        y_valid = torch.from_numpy(np.sin(x_valid.numpy())) 
        dataset_valid = SiameseTensorDataset(x_valid.float(), x_valid.float(), y_valid.float())
        
        x = torch.from_numpy(np.random.normal(0, 1, 10000).reshape(10000,1))
        y = torch.from_numpy(np.sin(x.numpy())) 
        dataset = SiameseTensorDataset(x.float(), x.float(), y.float())
        dataloader_training = DataLoader(dataset, batch_size=10, shuffle=True, sampler=None, batch_sampler=None)
        
        model = Net()
        criterion = nn.MSELoss()

        # CREATE A TRAINER
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        my_trainer = SiameseTrainer(Logger(logdir='', log_batch_interval=500), model, criterion, 
            optimizer, model_filename="testcase",  model_keep_epochs=False)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_training,
                                        epochs=4,
                                        validation_set=dataset_valid, 
                                        eval_interval=1)
        assert np.allclose(trainingsstate["training_loss[]"][-1][1], 0.034056082367897172, 1e-5)
        assert trainingsstate["training_loss[]"][-1][1] < trainingsstate["training_loss[]"][0][1]
        
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
                os.remove(os.path.join(dir,file))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--test', default="", metavar="",
                        help="Execute a specific test")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]
    if argv.test is not "":
        eval(str("TestUmmon()." + argv.test + '()'))
    else:
        unittest.main()
