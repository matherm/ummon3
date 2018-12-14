#############################################################################################
# Append the path to ummon3 to PATH-Variable so that ummon can be imported during development
import sys
sys.path.insert(0,'../../ummon3')  
sys.path.insert(0,'../ummon3')     
#############################################################################################

import unittest
import os
import numpy as np
from torchvision import transforms
from ummon import *

# set fixed seed for reproducible results
torch.manual_seed(4)
np.random.seed(4)

class TestUmmonFeatures(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestUmmonFeatures, self).__init__(*args, **kwargs)
        
        # BACKUP files
        backup_dir = "__backup__"
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
        y =  FeatureCache(VGG19Features(),cachedir="./")(x)        
        y =  FeatureCache(VGG19Features(),cachedir="./", clearcache=False)(x)        
        FeatureCache(VGG19Features(),cachedir="./")
        assert y.size(0) == 512 and y.size(1) == 2 and y.size(2) == 2
        
        # TEST Gram
        y =  FeatureCache(VGG19Features(gram=True),cachedir="./")(x)        
        y =  FeatureCache(VGG19Features(gram=True),cachedir="./", clearcache=False)(x)
        FeatureCache(VGG19Features(gram=True),cachedir="./")
        assert y.size(0) == 512 == y.size(1)

        # TEST random init
        y =  VGG19Features(gram=True, pretrained=False)(x)
        assert y.size(0) == 512 == y.size(1)
        
        
    def test_portilla_simoncelli_features(self):

        from ummon.features.psTMfeatures import PSTMfeatures
        # PREPARE DATA
        x = torch.from_numpy(np.random.uniform(0, 1, 32*32).reshape(32,32))

        # TEST EXTRACTOR
        y = PSTMfeatures(scales=2)(x)
        y = PSTMfeatures(scales=2, inclPixelStats=False)(x)


    def test_portilla_simoncelli_color_features(self):

        from ummon.features.psTMColorfeatures import PSTMColorfeatures
        # PREPARE DATA
        x = torch.from_numpy(np.random.uniform(0, 1, 32*32*3).reshape(32,32,3))

        # TEST EXTRACTOR
        y = PSTMColorfeatures(scales=2)(x)
        y = PSTMColorfeatures(scales=2)(x)

    def test_swEVM_features(self):

        from ummon.features.swEVMfeatures import swEVMfeatures

        # Load test data from ML implementation
        import scipy.io as sio

        filename = 'input.mat'
        object_name = 'input'
        sw_mat = sio.loadmat(str('ummon/tests/testdata/sw_evm/' + filename))
        input = sw_mat[object_name]

        # Test matlab implementation output      
        filename = 'outputDecomp.mat'
        object_name = 'out'
        sw_mat = sio.loadmat(str('ummon/tests/testdata/sw_evm/' + filename))
        outputDecomp_ml = np.asarray(np.abs(sw_mat[object_name]).flatten(), dtype='float32')
        
        filename = 'outputNormalized.mat'
        object_name = 'outputNormalized'
        sw_mat = sio.loadmat(str('ummon/tests/testdata/sw_evm/' + filename))
        outputNormalized_ml = np.asarray(sw_mat[object_name], dtype='float32')
        outputNormalized_ml_mean = np.mean(np.mean(outputNormalized_ml, 0), 0).flatten()

        # Test without pooling
        pyr = swEVMfeatures(normalized=False, pooling_mode='None')(input)
        pyr_normalized = swEVMfeatures(normalized=True, pooling_mode='None')(input)
        pyr_normalized_mean = swEVMfeatures(normalized=True, pooling_mode='mean_freq_orient')(input)

        assert (np.allclose(pyr.data.numpy(), outputDecomp_ml, rtol=1e-05, atol=1e-5))
        assert (np.allclose(pyr_normalized.data.numpy(), outputNormalized_ml.flatten()))
        assert (np.allclose(pyr_normalized_mean.data.numpy(), outputNormalized_ml_mean, rtol=1e-05, atol=1e-5))

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
