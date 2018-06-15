import torch
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from pathlib import Path
import paramiko
import getpass
import tarfile

__all__ = ["CuretVGG19grams"]

class CuretVGG19grams(Dataset):
    """
    Processed Curet dataset.
    Dataset is downloaded from ios share. Hence, you need an account on iosds01.
    
    Features: ['pool4', 'pool1', 'pool2', 'relu1_1', 'pool3', 'relu1_2']
    
    Author: M. Hermann
    """
    
    def __init__(self, download=True, train=True, path="/ext/data/curet_vgg19_grams", features="pool4", onehot=False):
     
        available_features = ['pool4', 'pool1', 'pool2', 'relu1_1', 'pool3', 'relu1_2']
        assert features in available_features
        
        self.DOWNLOAD_PATH = '/img-data/CUReT_VGG19_GRAMS/curet_vgg19_grams.tar.gz'
        self.host = "iosds01.ios.htwg-konstanz.de"
        self.port = 22
        
        self.base_path = path
        self.full_path = path + ""
        self.download = download
        self.onehot = onehot
        self.features = features
        self.train = train
        
        self.train_percentage = 0.95
        
        if download and not os.path.exists(self.full_path):
            self.download_file(self.base_path)
        
        self.files = sorted(Path(self.full_path).glob('**/*.npz'))
        
        np.random.seed(44)
        self.shuffled_idx = np.random.permutation(len(self.files))
        
    
    def stats(self):
        return {
            "name"  : "CURET Dataset (VGG19-grams)",
            "data split" : self.train_percentage,
            "data set" : "train" if self.train else "test",
            "data samples": len(self.files),
            "data shape" : self.__getitem__(0)[0].shape,
            "data dtype" : self.__getitem__(0)[0].dtype,
            "data label example" : self.__getitem__(0)[1]
            }
    
    def __repr__(self):
        return str(self.stats())
    
    
    def download_file(self, path):
        """
        Python function to download and extract the dataset.
        Login with username and password, and check the path to extract the downloaded file.
        """
        # CONNECT        
        transport = paramiko.Transport((self.host, self.port))
        print("Download path is:", str(self.host + self.DOWNLOAD_PATH))
        username = input('Username\n')
        pswd = getpass.getpass('Password:\n')
        transport.connect(username=username, password=pswd)
        sftp = paramiko.SFTPClient.from_transport(transport)

        # DONWLOAD
        print("Downloading...")
        if os.path.exists(path) == False:
            os.mkdir(path)
        sftp.get(remotepath=self.DOWNLOAD_PATH, localpath=path + "/dl.tar.gz", callback=None)
        sftp.close()
        transport.close()
        
        # EXTRACT
        tar = tarfile.open(path + "/dl.tar.gz", 'r')
        print("Extracting...")
        tar.extractall(path)
        tar.close()
        
        # CLEANUP
        os.remove(path + "/dl.tar.gz")
    
    def __getitem__(self, index):
        if self.train == False:
             index = index + int(np.ceil(len(self.files) * self.train_percentage))
        
        index = self.shuffled_idx[index]
        data = np.load(self.files[index])[self.features]
        label = int(str(self.files[index]).split("/")[-1].split("-")[0])
        assert label > 0 and label < 61
        return (torch.from_numpy(data).unsqueeze(0), label)
    
    def __len__(self):
        if self.train:
            return int(np.ceil(len(self.files) * self.train_percentage))
        else:
            return int(np.floor(len(self.files) * (1 - self.train_percentage)))
    
    # convert in one-hot code
    def one_hot(self, j, ndigits=10):
        """
        Returns a 'ndigits'-dimensional unit vector with a 1.0 in the jth position and zeroes
        elsewhere.  This is used to convert a digit (0...9) into a corresponding desired
        output from the neural network.
        """
        e = np.zeros(ndigits)
        e[j] = 1
        return e