'''
ummon: IOS Neural Network Package
=================================

Authors: Matthias O. Franz, Michael Grunwald, Matthias Hermann

Ummon is a neural network library based on PyTorch. The package is available as::

    from ummon import *

'''
import torch
import torchvision
import socket
import numpy
import sys
from platform import platform
import ummon.__version__
import ummon.utils as uu
import ummon.logger
import ummon.analyzer
import ummon.trainer
import ummon.schedulers
import ummon.visualizer
import ummon.modules
import ummon.features
import ummon.functionals
import ummon.trainingstate

version = ummon.__version__.version
__version__ = version

def system_info():
        print("SYSTEM INFORMATION (", socket.gethostname(),")" )
        print("---------------------------------")
        print("Platform:", platform())
        print("ummon3:", version)
        print("Python:", sys.version.split('\n'))
        print("CuDNN:", torch.backends.cudnn.version())
        print("CUDA:", torch.version.cuda) 
        print("Torch:", torch.__version__)
        print("Torchvision",torchvision.__version__)
        print("Numpy:", numpy.version.version)
        numpy.__config__.show()
        print("---------------------------------")

from .schedulers import *
from .trainer import *
from .trainingstate import *
from .logger import *
from .analyzer import *
from .visualizer import *
from .predictor import *
from .modules.container import *
from .modules.linear import *
from .modules.loss import *
from .modules.flatten import *
from .modules.conv import *
from .modules.pooling import *
from .modules.dropout import *
from .modules.lrn import *
from .modules.batchnorm import *
from .modules.imgproc import *
from .modules.vgg19 import *
from .modules.gram import *
from .transformations.imagetransforms import *
from .features.psTMfeatures import *
from .features.swEVMfeatures import *
from .features.vgg19features import *
from .features.featurecache import *
from .datasets import *
