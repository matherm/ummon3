import torch
from torch._utils import _accumulate
from torch import randperm
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle
import os
from pathlib import Path
import paramiko
import getpass
import zipfile

__all__ = ["PrimPatchVoxel"]

class PrimPatchVoxel(Dataset):
    """
    VoxelDateset: used to load the voxel primitives patches. Around 50k data.
    It downloads (if you want) the voxelset from the ios share. you need a account on iosds02 for download.
    
    Author: P. Fiur
    """
    
    def __init__(self, train=True, download=False, path="./primpatchvoxel", debug=False, onehot=False):
        # Central download path, host id and port
        self.DOWNLOAD_PATH = '/3D/3d-data/primitives/normVoxel64.zip'
        self.host = "iosds02.ios.htwg-konstanz.de" # iosds02
        self.port = 22
        
        # Local path relative to CWD (current working directory)
        self.base_path = path
        self.full_path = path + "/norm_64"
        self.train = train
        self.download = download
        self.debug = debug
        self.onehot = onehot
        
        if download and not os.path.exists(self.full_path):
            self.download_file(self.base_path)
        
        self.files = sorted(Path(self.full_path).glob('**/*.p'))
        self.statarray = np.zeros(6, dtype=np.int32)
        self.classes = 6
        self.train_percentage = 0.95
        
        print("Anzahl Dateien: ", len(self))
    
    def stats(self):
        """
        Python function to get a statistic.
        Call this function before running the network.
        :return:  Statistic Array
        """
        for i in self.files:
            with open(str(i), 'rb') as f:
                unpickled = pickle.load(f)
                x = unpickled[2]
                
                if x == 0:
                    self.statarray[0] = self.statarray[0] + 1
                if x == 1:
                    self.statarray[1] = self.statarray[1] + 1
                if x == 2:
                    self.statarray[2] = self.statarray[2] + 1
                if x == 3:
                    self.statarray[3] = self.statarray[3] + 1
                if x == 4:
                    self.statarray[4] = self.statarray[4] + 1
                if x == 5:
                    self.statarray[5] = self.statarray[5] + 1
        return {
            "name"  : "PrimPatchVoxel",                
            "cone": self.statarray[0],
            "cylinder": self.statarray[1],
            "ellipsoid": self.statarray[2],
            "sphere": self.statarray[3],
            "torus": self.statarray[4],
            "plane": self.statarray[5],
            }
    
    def __repr__(self):
        return str(self.stats())
    
    
    def compute_input_statistics(self, x):
        
        if x == 0:
            self.statarray[0] = self.statarray[0] + 1
        if x == 1:
            self.statarray[1] = self.statarray[1] + 1
        if x == 2:
            self.statarray[2] = self.statarray[2] + 1
        if x == 3:
            self.statarray[3] = self.statarray[3] + 1
        if x == 4:
            self.statarray[4] = self.statarray[4] + 1
        if x == 5:
            self.statarray[5] = self.statarray[5] + 1
        
        if np.sum(self.statarray) == len(self):
            print("Max: ", np.amax(self.statarray))
            print("Min: ", np.amin(self.statarray))
            print("Avg: ", np.mean(self.statarray))
            print("Median: ", np.median(self.statarray))
            print("Sum: ", np.sum(self.statarray))
            
            print("distribution:")
            a = self.array_as_dictionarie(self)
            print(a)
    
    def download_file(self, path):
        """
        Python function to download and extract the dataset.
        Login with username and password, and check the path to extract the downloaded file.
        :return: zippath: path to the extraced files
        """
        
        transport = paramiko.Transport((self.host, self.port))
        
        print("Downloading from:", str(self.host + self.DOWNLOAD_PATH))
        username = input('Enter your username\n')
        pswd = getpass.getpass('Enter your password:\n')
        print("Connecting...")
        transport.connect(username=username, password=pswd)
        
        
        sftp = paramiko.SFTPClient.from_transport(transport)
        if os.path.exists(path) == False:
            os.mkdir(path)
        print("Downloading...")
        sftp.get(remotepath=self.DOWNLOAD_PATH, localpath=path + "/normVoxel64.zip", callback=None)
        
        sftp.close()
        transport.close()
        print("Download done.\n\n")
        
        print("Start extract file.")
        zip_ref = zipfile.ZipFile(path + "/normVoxel64.zip", 'r')
        print("Extracting...")
        zip_ref.extractall(path)
        zip_ref.close()
        os.remove(path + "/normVoxel64.zip")
        print("Done")
    
    def __getitem__(self, index):
        """
        Python function to get one item from the total dataset.
        :param index: index of the current file
        :return: dataset
        """
        if self.debug:
            print("Voxel-file: #", index, self.files[index])
        
        if self.train == False:
            index = index + int(np.ceil(len(self.files) * self.train_percentage))
        
        with open(str(self.files[index]), 'rb') as f:
            unpickled = pickle.load(f)
            X0 = np.array(unpickled[0].todense())[0]
            y0 = unpickled[2]
        if self.debug:
            self.compute_input_statistics(y0)
        if self.onehot:
            y0 = self.one_hot(y0, ndigits=self.classes)
        
        # X e [1, VOXEL_COUNT], y e [Classes]
        xyz = int(np.ceil(np.power(len(X0), 1 / 3)))
        return (torch.from_numpy(np.asarray(X0)).float().view((1, xyz, xyz, xyz)), y0)
    
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