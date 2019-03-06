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
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
from torch.autograd import Variable
from torch.utils.data.dataset import TensorDataset
from torch.utils.data import DataLoader
import ummon.utils as uu
from ummon.analyzer import *
from ummon.modules.container import *
from ummon.trainingstate import *
from ummon.datasets import *

from torchvision import transforms

# set fixed seed for reproducible results
torch.manual_seed(4)
np.random.seed(4)

def flatten_transform(tensor):
    return tensor.contiguous().view(-1).clone()

class TestDatasets(unittest.TestCase):
    
    def __init__(self, *args, **kwargs):
        super(TestDatasets, self).__init__(*args, **kwargs)
        
        # BACKUP files
        backup_dir = "__backup__"
        files = os.listdir(".")
        dir = "."
        for file in files:
            if file.endswith(Trainingstate().extension) or file.endswith(".log"):
                if not os.path.exists(backup_dir):
                    os.makedirs(backup_dir) 
                os.rename(os.path.join(dir,file), os.path.join(backup_dir,file))
    
 
    def test_in_ram_numpy_data_set(self):

        patches_reference = TensorDataset(torch.zeros(1000).view(10,10,10), torch.zeros(10))
        data_point = patches_reference[0]
        assert LoadDatasetIntoRam(patches_reference).data.shape[0] == 10
        assert LoadDatasetIntoRam(patches_reference).labels.shape[0] == 10


    def test_pre_transform_dataset(self):
        transform = transforms.Compose(
            [transforms.Normalize(mean=[0., 0., 0.], std=[1.0, 1.0, 1.0]), flatten_transform]
            )
        path = "__ummoncache__/test/pretransformed"
        data = torch.from_numpy(np.random.normal(0,1,size=10*10*10*3)).float().view(10, 3,10,10)
        patches_reference = TensorDataset(data, torch.zeros(10))

        ds_1 = PreTransformDataset(patches_reference, transform, workers=2, path=path)
        ds_2 = PreTransformDataset(patches_reference, transform, workers=1, path=path)
        [os.remove(os.path.join(path, f)) for f in os.listdir(path)]
        assert np.allclose(ds_1[0][0].numpy(), ds_2[0][0].numpy(), rtol=0, atol=1e-5)

        ds_2 = PreTransformDataset(patches_reference, transform, workers=1, path=path)
        ds_1 = PreTransformDataset(patches_reference, transform, workers=10, path=path)
        [os.remove(os.path.join(path, f)) for f in os.listdir(path)]
        assert np.allclose(ds_1[0][0], ds_2[0][0], rtol=0, atol=1e-5)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('test', default="", metavar="",
                        help="Execute a specific test")
    argv = parser.parse_args()
    sys.argv = [sys.argv[0]]
    if argv.test is not "":
        eval(str("TestDatasets()." + argv.test + '()'))
    else:
        unittest.main()
