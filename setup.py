import os
import time
import subprocess
import sys
import socket
from setuptools import setup
from setuptools.command.install import install
from setuptools.command.test import test
from setuptools import find_packages
import numpy
import numpy.distutils.system_info as sysinfo
try:
    import torch
except:
    print("No `torch` found. Exiting..")
    exit()


"""
ummon3: IOS Neural Network Package
--------------------------------------------

This setup py is used to prepare everything for use.

@author Matthias O. Franz, Matthias Heramnn, Michael Grunwald, Pascal Laube
"""

import ummon.__version__
__version__ = ummon.version

class PreInstallation(install):
    """
    Installs needed local packages and runs some pre-installation procedures
    """
    def run(self):
        install.run(self)
        print("")
        print("########################")
        print("#   WELCOME to ummon3")
        print("#  ",__version__)
        print("########################\n")
        self.print_info()
        cwd = os.getcwd()
        #os.chdir("./utils/eigency/")
        #subprocess.call(
        #        "python setup.py install", shell=True
        #)
        print("\n To test your installation:")
        print("\tpython setup.py test")
        print("\n To test your performance:")
        print("\tpython setup.py performance")
        print("\n To check version:")
        print("\tpython setup.py --version")
        print("\n To run an example:")
        print("\tpython examples/basicusage.py")
        print("\n To import and use ummon:")
        print("\timport ummon")
        print("\tprint(ummon.version)")


    def print_info(self):
        print("SYSTEM INFORMATION (", socket.gethostname(),")" )
        print("---------------------------------")
        print("Python:", sys.version_info[0:3])
        print("CuDNN:", torch.backends.cudnn.version())
        print("CUDA:", torch.version.cuda) 
        print("Torch:", torch.__version__)
        print("Numpy:", numpy.version.version)
        numpy.__config__.show()
        print("---------------------------------")

class TestPerformance(install):
    def run(self):
        print("\nStarting performance tests..\n")
        self.test_cpu()
        if torch.cuda.is_available():
            self.test_pci()
            self.test_cuda()
        else:
            print("\nWARNING: CUDA is not available.")
        print("\nPerformance Test finished.\n")

    def test_pci(self):
        print("Test PCI-BUS")
        print("--------------------")
        CPU = torch.IntTensor(3000, 3000).zero_()
        torch.cuda.synchronize()
        t = time.time()
        for i in range(20):
            GPU = CPU.cuda()
            CPU = GPU.cpu()
        torch.cuda.synchronize()
        print("{0:.2f}".format(32*3000*3000*20*2/(1e9*(time.time() - t))), "Gbit/s")

    def test_cpu(self):
        print("Test CPU-Throughput")
        print("--------------------")
        SEED =  torch.from_numpy(numpy.random.randn(10000,10000)).float()
        CPU =  torch.FloatTensor(10000, 10000).zero_() + SEED
        t = time.time()
        CPU = CPU.matmul(CPU)
        print(int((10000)**2.80/(1e9*(time.time() - t))), "GFLOPS (FP32 MUL) BENCHMARK INFO: i7-7700: 8.4 GFLOPS per Core") 

    def test_cuda(self):
        print("Test CUDA-Throughput")
        print("--------------------")
        SEED = torch.from_numpy(numpy.random.randn(10000,10000)).float()
        GPU =  (torch.FloatTensor(10000, 10000).zero_() + SEED).cuda()
        torch.cuda.synchronize()
        GPU = torch.matmul(GPU,GPU)
        torch.cuda.synchronize()
        t = time.time()
        GPU = torch.matmul(GPU,GPU)
        torch.cuda.synchronize()
        print(int((10000+10000-1)**2.9/(1e9*(time.time() - t))), "GFLOPS (FP32) BENCHMARK INFO: 1080TI: 10609, P100: 10600, K40: 4290, K20: 1200 ")


setup(
      name = 'ummon',
      author = 'Matthias O. Franz, Matthias Heramnn, Michael Grunwald, Pascal Laube',
      author_email = 'Matthias.Hermann@htwg-konstanz.de',
      version = __version__,
      install_requires=['numpy>=1.5.0'],
      python_requires='>=3.5',
      license = 'GNU Library or Lesser General Public License (LGPL)',
      platforms=["OS Independent"],
      keywords=['numpy', 'pytorch'],
      long_description= "",
      url="https://git.ios.htwg-konstanz.de/mof-applications/ummon3",
      classifiers=[
      'Development Status :: Beta',
      'Intended Audience :: Developers',
      'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
      'Operating System :: OS Independent',
      'Programming Language :: Python',
      ],
      test_suite="ummon.tests",
#      py_modules=[
#        '__version__',
#        'ummon/analyzer',
#        'ummon/logger',
#        'ummon/trainer',
#        'ummon/trainingstate',
#        'ummon/visualizer',
#        'ummon/modules/bspline',
#        ],
#     package_dir = {'ummon': '',
#                   'ummon.modules' : 'ummon/modules',},
     packages=['ummon', 'ummon.modules', 'ummon.tests'],
     setup_requires=[
        'numpy',
        'setuptools',
        'scipy'
     ],
      cmdclass={ 'install':  PreInstallation,
                 'performance' : TestPerformance}
  )