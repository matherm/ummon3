'''
This module downloads the original MNIST data set from the official web site and converts
it into numpy/sklearn format.

Author: M. Hermann, M.O.Franz
'''
import os, struct
import io
from array import array
import numpy as np
import urllib.request
import gzip


def read(digits, path = "."):
    """
    Python function for importing the MNIST data set::
    
        X0,y0,Xv,yv,X1,y1 = mnist.read([0,1,2,3,4,5,6,7,8,9],, path="")
    
    returns the 28 x 28 input images as rows of the int32 data matrix X with format n x 728 
    and the labels as n-vector of integers as class labels (n = 50.000 for training,
    n = 10.000 for validation and testing). X0,y0 ist the training, Xv, vy the validation
    and X1, y1 the test set.
    
    Parameters:
    
        * 'digits' is a list of the digit classes to be included. 
        * 'path' is the path where the dataset is stored or downloaded.
    
    Calls loader() to download MNIST if necessary. 
    """
    
    # download if necessary
    if not (os.path.exists(os.path.join(path, 'train-images-idx3-ubyte'))):
        path = loader()
    
    # file name of training image  and label file
    fname_img = os.path.join(path, 'train-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    
    # unpack train label file
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()
    
    # unpack train image file
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()
    
    # convert into numpy format
    ind = [ k for k in range(size) if lbl[k] in digits ]
    ndigits = len(digits)
    images =  np.zeros((len(ind), rows*cols))
    labels = np.zeros((len(ind), ndigits))
    for i in range(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i, :] = one_hot(lbl[ind[i]], ndigits)
    
    # convert to int32
    images = np.array(images, dtype="int32")
    labels = np.array(labels, dtype="int32")
    
    # divide into training and validation set
    X0 = images[:50000,:]
    Xv = images[50000:,:]
    y0 = labels[:50000]
    yv = labels[50000:]
    
    # file name of test image  and label file    
    fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
    fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    
    # unpack label file
    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = array("b", flbl.read())
    flbl.close()
    
    # unpack image file
    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = array("B", fimg.read())
    fimg.close()
    
    # convert into numpy format
    ind = [ k for k in range(size) if lbl[k] in digits ]
    images =  np.zeros((len(ind), rows*cols))
    labels = np.zeros((len(ind), ndigits))
    for i in range(len(ind)):
        images[i, :] = img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]
        labels[i, :] = one_hot(lbl[ind[i]], ndigits)
    
    # convert to int32
    X1 = np.array(images, dtype="int32")
    y1 = np.array(labels, dtype="int32")
    
    print("MNIST, Classes:     " + str(y0.shape[1]))
    print("Training Images:    " + str(X0.shape))
    print("Validation Images:  " + str(Xv.shape))
    print("Test Images:        " + str(X1.shape))
    
    return X0,y0,Xv,yv,X1,y1


# convert in one-hot code
def one_hot(j, ndigits=10):
    """
    Returns a 'ndigits'-dimensional unit vector with a 1.0 in the jth position and zeroes 
    elsewhere.  This is used to convert a digit (0...9) into a corresponding desired 
    output from the neural network.
    """
    e = np.zeros((ndigits))
    e[j] = 1.0
    return e


# download dataset
def loader():
    
    # cerate dir
    CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))
    PATH = os.path.join(CURRENT_PATH, 'MNIST') + "/"
    if not (os.path.exists(PATH)):
        os.makedirs(PATH)
    
    if not (os.path.exists(os.path.join(PATH, 'train-images-idx3-ubyte'))):
        print("Downloading MNIST ...")
        
        MNIST_URL = "http://yann.lecun.com/exdb/mnist/"
        FILE_TRAINING = "train-images-idx3-ubyte"
        FILE_TRAINING_Y = "train-labels-idx1-ubyte"
        FILE_TEST = "t10k-images-idx3-ubyte"
        FILE_TEST_Y = "t10k-labels-idx1-ubyte"
        
        # Training images
        response = urllib.request.urlopen(MNIST_URL + FILE_TRAINING + ".gz")
        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        with open(PATH + FILE_TRAINING , 'wb') as outfile:
            outfile.write(decompressed_file.read())
        
        # Training labels
        response = urllib.request.urlopen(MNIST_URL + FILE_TRAINING_Y + ".gz")
        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        with open(PATH + FILE_TRAINING_Y , 'wb') as outfile:
            outfile.write(decompressed_file.read())
        
        # Test images
        response = urllib.request.urlopen(MNIST_URL + FILE_TEST + ".gz")
        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        with open(PATH + FILE_TEST , 'wb') as outfile:
            outfile.write(decompressed_file.read())
        
        # test labels
        response = urllib.request.urlopen(MNIST_URL + FILE_TEST_Y + ".gz")
        compressed_file = io.BytesIO(response.read())
        decompressed_file = gzip.GzipFile(fileobj=compressed_file)
        with open(PATH + FILE_TEST_Y , 'wb') as outfile:
            outfile.write(decompressed_file.read())
    
    return PATH

