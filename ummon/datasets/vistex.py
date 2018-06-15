import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from pathlib import Path
from imageio import imread
import paramiko
from stat import S_ISDIR
import getpass
from PIL import Image
from os.path import basename
import requests 
import tarfile
import subprocess
import shutil

__all__ = ["VisTexReference"]
    
class VisTexReference(Dataset):
    """
    
    VisTex
    
    Dataset is downloaded from ios share. Hence, you need an account on iosds01.
    
    
    Author: M. Hermann
    """
    
    def __init__(self, download=True, train=True, path="/ext/data/vistex", onehot=False):
        
        self.WEBLINK = 'http://vismod.media.mit.edu/pub/VisTex/VisTex.tar.gz'
        self.VISTEXREFERENCE = "/Images/Reference"
        
        self.LABELS = ["Bark", "Brick", "Buildings", "Clouds", "Fabric", "Flowers", "Food", "Grass", "Leaves", "Metal", "Misc", "Painting", "Sand", "Stone", "Terrain", "Tile", "Water", "WheresWaldo", "Wood"]
        
        self.base_path = path
        self.vistex_reference_path = self.base_path + self.VISTEXREFERENCE
        self.download = download
        self.onehot = onehot
        self.train = train
        
        self.train_percentage = 0.95
        
        if download and (not os.path.exists(self.base_path) or len(os.listdir(self.base_path)) < 3):
            self.download_file(self.WEBLINK, self.base_path)
        
        self.files = sorted(Path(self.vistex_reference_path).glob('**/*.ppm'))
        
        np.random.seed(44)
        self.shuffled_idx = np.random.permutation(len(self.files))
    
    def stats(self):
        return {
            "name"  : "VisTex Dataset",
            "data split" : self.train_percentage,
            "data set" : "train" if self.train else "test",
            "data samples": len(self.files),
            "data shape" : self.__getitem__(0)[0].shape,
            "data dtype" : self.__getitem__(0)[0].dtype,
            "data label example" : self.__getitem__(0)[1]
            }
    
    def __repr__(self):
        return str(self.stats())


    def download_file(self, remote, path):
        """
        Python function to download and extract the dataset.
        Login with username and password, and check the path to extract the downloaded file.
        """
        # CONNECT        
        print("Downloading...")
        r = requests.get(remote)

        # DONWLOAD
        if os.path.exists(path) == False:
            os.mkdir(path)
    
        with open(path + "/dl.tar.gz", 'wb') as f:  
            f.write(r.content)
        
        # EXTRACT
        tar = tarfile.open(path + "/dl.tar.gz", 'r')
        print("Extracting...")
        tar.extractall(path)
        tar.close()
        
        # CLEANUP
        os.remove(path + "/dl.tar.gz")

        # INSTALL MAGIC
        cwd = os.getcwd()
        os.chdir(path + "/VisionTexture")
        subprocess.call(
                "sh buildVisTex", shell=True
        )
        os.chdir(cwd)
        os.rename(path + "/VisionTexture/VisTex/FLAT", path + "/FLAT")
        os.rename(path + "/VisionTexture/VisTex/Images", path + "/Images")
        os.rename(path + "/VisionTexture/VisTex/README", path + "/README")
        shutil.rmtree(path + "/VisionTexture")
            
    
    def __getitem__(self, index):
        if self.train == False:
             index = index + int(np.ceil(len(self.files) * self.train_percentage))
        
        index = self.shuffled_idx[index]
        data = np.asarray(Image.open(self.files[index])) # shape is Y x X x C
        data = np.transpose(data, (2,0,1))
        

        label_text = str(self.files[index]).split("/")[-1].split(".")[0]
        for i, l in enumerate(self.LABELS):
            if l == label_text:
                label = i
                break
        
        assert label >= 0 and label < 19
        if self.onehot:
            return (torch.from_numpy(data), self.one_hot(label - 1, 19))
        else:
            return (torch.from_numpy(data), label)
    
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