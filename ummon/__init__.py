'''
ummon: IOS Neural Network Package
=================================

Authors: Matthias O. Franz, Michael Grunwald, Matthias Hermann

Ummon is a neural network library based on PyTorch. The package is available as::

    from ummon import *

'''
from .__version__ import __version__

try:
    import torch
    import torchvision
    import socket
    import numpy
    import sys
    from platform import platform
    import ummon.utils
    import ummon.logger
    import ummon.analyzer
    import ummon.trainer
    import ummon.schedulers
    import ummon.visualizer
    import ummon.modules
    import ummon.features
    import ummon.functionals
    import ummon.trainingstate

    def system_info():
        print("SYSTEM INFORMATION (", socket.gethostname(), ")")
        print("---------------------------------")
        print("Platform:", platform())
        print("ummon3:", ummon.__version__)
        print("Python:", sys.version.split('\n'))
        print("CuDNN:", torch.backends.cudnn.version())
        print("CUDA:", torch.version.cuda)
        print("Torch:", torch.__version__)
        print("Torchvision", torchvision.__version__)
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
    from .metrics import *

except Exception as e:
    def system_info():
        print("SYSTEM INFORMATION not available")
        print("---------------------------------")
    print("Some ummon dependencies are not installed, ERROR: ", e)
