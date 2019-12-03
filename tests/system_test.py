# -*- coding: utf-8 -*-
import unittest
import torch
import numpy as np
import torchvision

class TestSystem(unittest.TestCase):

    def test_framework_system_environment(self):
        
        developer_tested_pytorch_versions = ["0.4.0", "0.4.1", "1.0.0", "1.2.0"]
        assert torch.__version__ in developer_tested_pytorch_versions
        
        developer_tested_numpy_versions =  ["1.13.3", "1.14.0", "1.14.3", "1.15.1", "1.16.3"]
        assert np.version.version in developer_tested_numpy_versions

        developer_tested_torchvision_versions =  ['0.2.1', "0.4.0", '0.4.0a0+9232c4a']
        assert torchvision.__version__ in developer_tested_torchvision_versions
        
    def test_cuda_system_environment(self):

        if not torch.cuda.is_available():
            print('\nWarning: cannot run this test - Cuda is not enabled on your machine.')
        return    
    
        developer_tested_cuda_versions = ["8.0.61" , "9.2.148"]
        assert torch.version.cuda in developer_tested_cuda_versions
        
        developer_tested_cudnn_versions = [7005,7102, 7602]
        assert torch.backends.cudnn.version() in developer_tested_cudnn_versions


if __name__ == '__main__':
    unittest.main()
