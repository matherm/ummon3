# ummon3: IOS Neural Network Package
Author: Matthias O. Franz, Matthias Hermann, Michael Grunwald, Pascal Laube

ummon3 is a neural network library written in Python, Numpy, Scipy and PyTorch.

## Dependencies
The package requires Python 3 and just needs the standard packages `numpy`, `scipy`, 
`setuptools` and `PyTorch` which can be installed on Anaconda [recommended] as

    conda create -n ummon3 python=3.6 anaconda 
    source activate ummon3
    conda install pytorch torchvision -c pytorch
	
On Windows there exist prebuilt libraries with cuda support packaged by peterjc123.
Install on Windows via

	conda install -c peterjc123 pytorch

or for Cuda 9 (Windows)

	conda install -c peterjc123 pytorch cuda90

To remove an existing environment

    conda remove --name myenv --all

To list all existing Anaconda environments

    conda env list

To install to an arbitrary location

    conda create -p /path/to/environment python=3.6 anaconda
    
To clone to an arbitrary location

    conda create -p /path/to/environment --clone `SOURCE`

or on a Mac (macOS Binaries dont support CUDA, install from source if CUDA is needed: https://github.com/pytorch/pytorch#from-source)

    pip3 install setuptools
    brew install openblas
    brew install numpy --with-openblas --with-python3
    brew install scipy --with-openblas --with-python3
    pip3 install http://download.pytorch.org/whl/torch-0.3.1-cp35-cp35m-macosx_10_6_x86_64.whl 
    pip3 install torchvision
    pip3 install scipy  
    

or in linux

    apt-get install libopenblas-dev
    apt-get install python3-setuptools python3-numpy python3-scipy
    pip3 install http://download.pytorch.org/whl/torch-0.3.1-cp35-cp35m-macosx_10_6_x86_64.whl 
    pip3 install torchvision
    pip3 install scipy
	
or Windows

	conda install numpy scipy setuptools scikit-learn matplotlib
	pip install torchvision

If you want to run the example and test code, you need additional Python packages
(`scikit-learn` and `matplotlib`). On Anaconda

    source activate ummon3
    conda install scikit-learn matplotlib


On MacOsX

    brew install matplotlib --with-python3
    pip3 install sklearn

In linux

    apt-get install python3-sklearn python3-matplotlib

If you want to create the documentation, you need `doxygen`, `graphviz` and `sphinx`:

on Anaconda

    apt-get install doxygen
    source activate ummon3
    conda install sphinx graphviz

on MacOsx

    brew install doxygen
    pip3 install sphinx graphviz

or

    apt-get install doxygen graphviz python3-sphinx
	
or on Windows

	conda install -c conda-forge doxygen 
	conda install sphinx graphviz

## Using the GPU
To use the GPU, Nvidia’s GPU-programming toolchain is required. You should install at least the CUDA driver and the CUDA Toolkit.
PyTorch comes with its own CUDA-binaries (unfortunately not on MacOSX). On Linux do

Check driver version:

    nvidia-smi

Install appropriate cuda-version: e.g. for version 375 install https://developer.nvidia.com/compute/cuda/8.0/Prod2/local_installers/cuda_8.0.61_375.26_linux-run
Update with patch: https://developer.nvidia.com/compute/cuda/8.0/Prod2/patches/2/cuda_8.0.61.2_linux-run

Install the Cuda-Samples during installation:

    cd ./NVIDIA_CUDA-8.0_Samples
    make
    cd ./bin/x86_64/linux/release/

execute tests:

    ./deviceQuery
    ./bandwidthTest

On MacOsx

    Follow: https://github.com/pytorch/pytorch#from-source

To test your GPU support within OSX do the following:

To test your GPU support, cut and paste the following program into a file and run it.

    https://git.ios.htwg-konstanz.de/mathebox/publishing/blob/master/cuda-pytorch-test/test_pytorch_cuda.py

## Installation
Clone ummon from the IOS Github via

For latest development version:
    git clone git@git.ios.htwg-konstanz.de:mof-applications/ummon3.git
    
For latest stable version (replace TAG with e.g. 3.4.0)
    git clone -b `TAG` git@git.ios.htwg-konstanz.de:mof-applications/ummon3.git

Go into the installation directory and execute:

    cd ummon3
    python3 setup.py install

in Anaconda

    cd ummon3
    source activate ummon3
    python setup.py install

Test your installation by calling 

    python3 setup.py test
    
Test your performance by calling 

    python3 setup.py performance

The module should be available as:

    import ummon
    ummon.system_info()

## Start using ummon
Validate your installation:

    import ummon.tests
    ummon.tests.validation()

Test your system performance:

    import ummon.tests
    ummon.tests.performance()

## Directory structure
* examples: examples of use
* ummon: python package directory
* ummon/tests : tests
* ummon/modules: ios extensions to pytorch