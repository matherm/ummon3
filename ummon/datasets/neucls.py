import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
from pathlib import Path
from imageio import imread
import paramiko
import getpass
import tarfile

__all__ = ["NEUCLS"]

class NEUCLS(Dataset):
    """
    Dataset is downloaded from ios share. Hence, you need an account on iosds01.
    
    Originally taken from:
        http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html
        
    This dataset is used by Silicon Software.
    
    Author: M. Hermann
    """
    
    def __init__(self, download=True, train=True, path="/ext/data/neucls", onehot=False):
     
        self.LABELS = ["Sc", "RS", "PS", "Pa", "In", "Cr"]
        
        self.DOWNLOAD_PATH = '/img-data/NEU-CLS/NEU-CLS.tar.gz'
        self.host = "iosds01.ios.htwg-konstanz.de"
        self.port = 22
        
        self.base_path = path
        self.full_path = path + ""
        self.download = download
        self.onehot = onehot
        self.train = train
        
        self.train_percentage = 0.95
        
        if download and (not os.path.exists(self.base_path) or len(os.listdir(self.base_path)) < 3):
            self.download_file(self.base_path)
        
        self.files = sorted(Path(self.full_path).glob('**/*.bmp'))
        
        np.random.seed(44)
        self.shuffled_idx = np.random.permutation(len(self.files))
        
    
    def stats(self):
        return {
            "name"  : "NEU-CLS Dataset",
            "data split" : self.train_percentage,
            "data set" : "train" if self.train else "test",
            "data samples": len(self),
            "total samples" : len(self.files),
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
        data = imread(self.files[index])
        
         # Normalize
        data = data.astype(np.float32)
        data = data / 255.
        
        label_text = str(self.files[index]).split("/")[-1].split("_")[0]
        for i, l in enumerate(self.LABELS):
            if l == label_text:
                label = i
                break
            
        assert label >= 0 and label < 6
        if self.onehot:
            return (torch.from_numpy(data).unsqueeze(0), self.one_hot(label, 6))
        else:
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