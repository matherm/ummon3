import unittest
from math import log
import os

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
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
    
    # test fully connected layer
    def test_predict(self):
        print('\n')
        
        # create net
        cnet = Sequential(
            ('line0', Linear([5], 7, 'xavier_uniform', 0.001)),
            ('sigm0', nn.Sigmoid()),
            regression=True
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
        
        # test pytorch combined softmax and bce
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
    
    
    # test fit
    def test_logger(self):
        print('\n')
        
        with Logger('ummon', 10) as lg: # create net
            lg.debug('Test debugging output.')
            lg.warn('Test warning!')
            try:
                lg.error('Test error!', ValueError)
            except ValueError:
                print("Only a test - no worries ...")

    def test_trainer(self):
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
        my_trainer = Trainer(Logger2( logfile = "test.log", log_batch_interval=500), model, criterion, optimizer, model_filename="testcase", regression=True, model_keep_epochs=True)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        validation_set=dataset_valid, 
                                        epochs=5,
                                        eval_interval=2, 
                                        early_stopping=False)
        
        self.assertTrue(np.allclose(0.5037568211555481,trainingsstate.state["best_validation_loss"][1]))

        # RESTORE STATE
        my_trainer = Trainer(Logger2( logfile = "test.log", log_batch_interval=500), model, criterion, optimizer, model_filename="testcase", trainingstate=trainingsstate, regression=True, precision=np.float32)
        
        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        validation_set=dataset_valid, 
                                        epochs=1,
                                        eval_interval=2, 
                                        early_stopping=False)
        
        os.remove("test.log")
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(trainingsstate.extension):
                os.remove(os.path.join(dir,file))
        
    def test_trainer_cuda(self):
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
        my_trainer = Trainer(Logger2( logfile = "test.log", log_batch_interval=500), model, criterion, optimizer, model_filename="testcase", regression=True, model_keep_epochs=True, use_cuda=True)
        
        # START TRAINING
        trainingsstate = my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        validation_set=dataset_valid, 
                                        epochs=5,
                                        eval_interval=2, 
                                        early_stopping=False)

        self.assertTrue(np.allclose(0.5051723122596741,trainingsstate.state["best_validation_loss"][1]))

        # RESTORE STATE
        my_trainer = Trainer(Logger2( logfile = "test.log", log_batch_interval=500), model, criterion, optimizer, model_filename="testcase", trainingstate=trainingsstate, regression=True, precision=np.float32, use_cuda=True)

        # RESTART TRAINING
        my_trainer.fit(dataloader_training=dataloader_trainingdata,
                                        validation_set=dataset_valid, 
                                        epochs=1,
                                        eval_interval=2, 
                                        early_stopping=False)
        
        os.remove("test.log")
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
                     validation_batchsize = 0,
                     regression = True,
                     args = { "args" : 1 , "argv" : 2})
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = 0, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     validation_batchsize = 0,
                     regression = True,
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
        
        criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(Net().parameters(), lr=0.01)
        
        ts = Trainingstate()
        ts.update_state(0, Net(), criterion, optimizer, 0, 
                     validation_loss = 0, 
                     training_accuracy = 0,
                     training_batchsize = 0,
                     validation_accuracy = 0, 
                     validation_batchsize = 0,
                     args = { "args" : 1 , "argv" : 2})
        ts.save_state("test.pth.tar")
        ts.load_state("test.pth.tar")
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(ts.extension):
                os.remove(os.path.join(dir,file))
       
    
    def test_analyzer(self):
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
        Analyzer().inference(model, dataset_valid)
       
        pass
    
    def test_logger2(self):
        Logger2("test.log").info("Testlog")
        os.remove("test.log")
        
    
    def test_visualizer(self):
        pass
    
    
        
        


if __name__ == '__main__':
    unittest.main()
