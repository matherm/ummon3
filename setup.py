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
import torch


"""
ummon3: IOS Neural Network Package
--------------------------------------------

This setup py is used to prepare everything for use.

@author Matthias O. Franz, Matthias Heramnn, Michael Grunwald, Pascal Laube
"""

import ummon
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
        ummon.system_info()

import ummon.tests
class TestPerformance(install):
    def run(self):
        ummon.tests.performance()
        
setup(
      name = 'ummon',
      version = __version__,
      description= "ummon v3 is a neural network library written in Python, Numpy, Scipy and PyTorch.",
      author = 'Matthias O. Franz, Matthias Heramnn, Michael Grunwald, Pascal Laube',
      author_email = 'Matthias.Hermann@htwg-konstanz.de',
      keywords=['numpy', 'pytorch'],
      license = 'GNU Library or Lesser General Public License (LGPL)',
      url="https://git.ios.htwg-konstanz.de/mof-applications/ummon3",
      classifiers=[
          'Development Status :: Beta',
          'Intended Audience :: Developers',
          'License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)',
          'Operating System :: OS Independent',
          'Programming Language :: Python',
      ],
      platforms=["OS Independent"],
      install_requires=['numpy>=1.5.0'],
      python_requires='>=3.5',
      setup_requires=[
        'numpy',
        'setuptools',
        'scipy'
        ],
      test_suite="ummon.tests",
      packages=['ummon', 'ummon.modules', 'ummon.tests'],
      cmdclass={ 'install':  PreInstallation,
                 'performance' : TestPerformance}
  )