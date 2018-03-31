'''
ummon: IOS Neural Network Package
=================================

Authors: Matthias O. Franz, Michael Grunwald, Matthias Hermann, Martin Miller

Ummon is a neural network library based on PyTorch. The package is available as::

    import ummon

This documentation can be generated by typing in the subdirectory doc::

    make doc

'''
import torch
import socket
import numpy
import sys
import ummon.__version__
version = ummon.__version__.version

def system_info():
        print("SYSTEM INFORMATION (", socket.gethostname(),")" )
        print("---------------------------------")
        print("Python:", sys.version_info[0:3])
        print("CuDNN:", torch.backends.cudnn.version())
        print("CUDA:", torch.version.cuda) 
        print("Torch:", torch.__version__)
        print("Numpy:", numpy.version.version)
        numpy.__config__.show()
        print("---------------------------------")


# import files from package
from .modules.container import *
from .modules.linear import *
from .logger import Logger