'''
This module downloads the original CIFAR10 data set from the official web site and converts
it into numpy/sklearn format.

Author: M. Hermann, M.O.Franz
'''
import os, struct
import io
from array import array
import numpy as np
import urllib.request
import io
import tarfile
import pickle


def read(path = "."):
    """
    Python function for importing the CIFAR10 data set::
    
        X0,y0,X1,y1,X2,y2 = load_cifar10.read(path)
    
    returns the 3 x 32 x 32 input images as rows of the int32 data matrix X with format n x 3072 
    and the labels as n-vector of integers as class labels (n=45.500 for training,
    n = 4.500 for validation and testing). X0,y0 ist the training, X1, y1 the test
    and X2, y2 the validation set.
    
    Parameters:
    
        * 'path' is the path where the dataset is stored or downloaded.
    
    Calls loader() to download CIFAR10 if necessary. 
    """
    
    # download if necessary
    if not (os.path.exists(os.path.join(path, 'data_batch_1'))):
        path = loader()
    
    # training dataset
    X0 = []
    y0 = []
    for i in range(1,6,1):
        fo = open(str(path + 'data_batch_' + str(i)), 'rb')
        dict = pickle.load(fo, encoding="latin1")
        fo.close()
        arr = np.asarray(dict['labels'])
        arr = arr.reshape((arr.shape[0],1))
        if len(X0) == 0:
            X0 = dict['data']
            y0 = arr
        else:
            X0 = np.concatenate((X0, dict['data']), axis=0)
            y0 = np.concatenate((y0, arr), axis=0)
    
    # convert to int32
    X0 = np.array(X0, dtype="int32")
    y = np.array(y0, dtype="int32")
    
    # convert labels to one-hot coding
    y0 = np.zeros((len(y),10))
    for i in range(0, len(y)):
        y0[i,:] = one_hot(y[i], 10)
    
    # divide into training and validation set
    X2 = X0[45500:,:]
    y2 = y0[45500:,:]
    X0 = X0[:45500,:]
    y0 = y0[:45500,:]
    
    # test dataset
    fo = open(str(path + 'test_batch'), 'rb')
    dict = pickle.load(fo, encoding="latin1")
    fo.close()
    arr = np.asarray(dict['labels'])
    arr = arr.reshape((arr.shape[0],1))
    y1 = arr
    X1 = dict['data']
    
    # convert to int32
    X1 = np.array(X1, dtype="int32")
    y = np.array(y1, dtype="int32")

    # convert labels to one-hot coding
    y1 = np.zeros((len(y),10))
    for i in range(0, len(y)):
        y1[i,:] = one_hot(y[i], 10)
    
    print("CIFAR10, Classes:   " + str(y0.shape[1]))
    print("Training Images:    " + str(X0.shape))
    print("Test Images:        " + str(X1.shape))
    print("Validation Images:  " + str(X2.shape))
    
    return X0,y0,X1,y1,X2,y2


# convert in one-hot code
def one_hot(j, nclasses=10):
    """
    Returns a 'nclasses'-dimensional unit vector with a 1.0 in the jth position and zeroes 
    elsewhere.  This is used to convert a class label into a corresponding desired 
    output for the neural network.
    """
    e = np.zeros((nclasses))
    e[j] = 1.0
    return e


# download dataset
def loader():
    
    # cerate dir
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    PATH = os.path.join(CURRENT_PATH, 'CIFAR10') + "/"
    if not (os.path.exists(PATH)):
        os.makedirs(PATH)
    
    if not (os.path.exists(os.path.join(PATH + "cifar-10-batches-py", 'data_batch_1'))):
        print(os.path.join(PATH + "cifar-10-batches-py", 'data_batch_1'))        
        print("Downloading CIFAR10 ...")
        
        CIFAR10_URL = "http://www.cs.toronto.edu/~kriz/"
        FILE_NAME = "cifar-10-python"
        
        # Download
        response = urllib.request.urlopen(CIFAR10_URL + FILE_NAME + ".tar.gz")
        compressed_file = io.BytesIO(response.read())
        TarFile =  tarfile.open(fileobj=compressed_file, mode='r:gz')
        
        # Extract
        TarFile.extractall(path=PATH)
    
    return PATH + "/" + "cifar-10-batches-py" + "/"

