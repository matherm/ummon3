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

class TestUmmonFeatures(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestUmmonFeatures, self).__init__(*args, **kwargs)
        
        # BACKUP files
        backup_dir = "_backup"
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension) or file.endswith(".log"):
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir)
                os.rename(os.path.join(dir,file), os.path.join(backup_dir,file))
        

    def test_vgg19_gram_features(self):

        # PREPARE DATA
        x = torch.zeros(3*32*32).reshape(3,32,32)
        
        # TEST EXTRACTOR
        y = VGG19Features(cachedir="./")(x)        
        y = VGG19Features(cachedir="./", clearcache=False)(x)        
        VGG19Features(cachedir="./")
        assert y.size(0) == 512 and y.size(1) == 2 and y.size(2) == 2
        
        # TEST Gram
        y = VGG19Features(cachedir="./", gram=True)(x)        
        y = VGG19Features(cachedir="./", gram=True, clearcache=False)(x)        
        VGG19Features(cachedir="./", gram=True)
        
        assert y.size(0) == 512 == y.size(1)
        
        
    def test_image_patches_data_set(self):
        
        my_transforms = transforms.Compose([SquareAnomaly(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                mode='gray', \
                                transform=my_transforms) 
        y = test_set[0][0]
        
        assert y.size(0) == 1 and y.dim() == 3
        assert y.min() >= -10
        assert y.max() <= 10
        
        my_transforms = transforms.Compose([transforms.ToTensor()])
        test_set = AnomalyImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
        
        y = test_set[0][0]
        
        assert y.size(0) == 3 and y.dim() == 3
        assert y.min() >= 0
        assert y.max() <= 10
        
        
        my_transforms = transforms.Compose([transforms.ToTensor()])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms) 
        
        y = test_set[0][0]
        
        assert y.size(0) == 3 and y.dim() == 3
        assert y.min() >= 0
        assert y.max() <= 10
        
        
        my_transforms = transforms.Compose([TurtleAnomaly(pixels=16*16//8, thickness = 8, color_bucket=[0, 128, 199])])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                window_size=512,
                                train=False, \
                                transform=my_transforms) 
        
        assert test_set[0][0].min() == 0  
        
        
    def test_portilla_simoncelli_features(self):

        from ummon.preprocessing.psTMfeatures import PSTMfeatures
        # PREPARE DATA
        x = torch.from_numpy(np.random.uniform(0, 1, 32*32).reshape(32,32))

        # TEST EXTRACTOR
        y = PSTMfeatures()(x)
        print(y)

    def test_swEVM_features(self):

        from ummon.preprocessing.swEVMfeatures import swEVMfeatures

        # Load test data from ML implementation
        import scipy.io as sio

        filename = 'wood-0035_p1_gray'
        object_name = 'input'
        sw_mat = sio.loadmat(str('ummon/datasets/testdata/' + filename))
        img = sw_mat[object_name]

        ## Test extractor
        # decomp Gabor
        pyr = swEVMfeatures(normalized=False, meanFreqOutput=False)(img)

        ## Load reference output from ML implementation
        filename = 'evm_decomp_gabor_output_wood-0035_p1_gray'
        object_name = 'output'
        ml_obj_pyr = sio.loadmat(str('ummon/datasets/testdata/' + filename + '.mat'))
        ml_pyr = np.abs(ml_obj_pyr[object_name]).astype('float32').flatten()

        assert (np.allclose(pyr.data.numpy(), ml_pyr, rtol=1e-05, atol=1e-5))

              

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('test', default="", metavar="",
                        help="Execute a specific test")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]
    if argv.test is not "":
        eval(str("TestUmmonFeatures()." + argv.test + '()'))
    else:
        unittest.main()
