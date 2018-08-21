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
import ummon.supervised
import ummon.unsupervised
import ummon.schedulers
import ummon.trainingstate
import ummon.visualizer
import ummon.modules
import ummon.preprocessing
import ummon.functionals
import ummon.tests
import ummon.tools

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
from .trainingstate import *
from .trainer import *
from .unsupervised import *
from .supervised import *
from .logger import *
from .trainingstate import *
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
from .modules.vgg19 import *
from .modules.gram import *
from .gridsearch import *
from ummon.preprocessing.imagetransforms import *
from ummon.preprocessing.psTMfeatures import *
from ummon.preprocessing.swEVMfeatures import *
from ummon.preprocessing.vgg19features import *
from ummon.preprocessing.anomaly import *
from .datasets.generic import *

