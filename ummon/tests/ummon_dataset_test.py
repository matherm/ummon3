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
from ummon.datasets.generic import *

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
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), flatten_transform]
            )

        patches_reference = TensorDataset(torch.zeros(1000).view(10,10,10), torch.zeros(10))
        data_point = patches_reference[0]
        assert LoadDatasetIntoRam(patches_reference).data.shape[0] == 10
        assert LoadDatasetIntoRam(patches_reference).labels.shape[0] == 10


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
