import unittest
import ummon
import os
import torch
import time
import numpy

def validation():
    """
    Finds all ummon tests within the directory.
    Tests are identified by `*_test.py` file pattern.
    """
    ummon_directory = os.path.dirname(ummon.__file__)
    suite = unittest.TestLoader().discover(ummon_directory, pattern="*_test.py")
    unittest.TextTestRunner(verbosity=2).run(suite)
    
def performance():
    def test_pci():
        print("Test PCI-BUS")
        print("--------------------")
        CPU = torch.IntTensor(3000, 3000).zero_()
        torch.cuda.synchronize()
        t = time.time()
        for i in range(20):
            GPU = CPU.to('cuda')
            CPU = GPU.to('cpu')
        torch.cuda.synchronize()
        print("{0:.2f}".format(32*3000*3000*20*2/(1e9*(time.time() - t))), "Gbit/s")

    def test_cpu():
        print("Test CPU-Throughput")
        print("--------------------")
        SEED =  torch.from_numpy(numpy.random.randn(10000,10000)).float()
        CPU =  torch.FloatTensor(10000, 10000).zero_() + SEED
        t = time.time()
        CPU = CPU.matmul(CPU)
        print(int((10000)**2.80/(1e9*(time.time() - t))), "GFLOPS (FP32 MUL) BENCHMARK INFO: i7-7700: 8.4 GFLOPS per Core") 

    def test_cuda():
        print("Test CUDA-Throughput")
        print("--------------------")
        SEED = torch.from_numpy(numpy.random.randn(10000,10000)).float()
        GPU =  (torch.FloatTensor(10000, 10000).zero_() + SEED).to('cuda')
        torch.cuda.synchronize()
        GPU = torch.matmul(GPU,GPU)
        torch.cuda.synchronize()
        t = time.time()
        GPU = torch.matmul(GPU,GPU)
        torch.cuda.synchronize()
        print(int((10000+10000-1)**2.9/(1e9*(time.time() - t))), "GFLOPS (FP32) BENCHMARK INFO: 1080TI: 10609, P100: 10600, K40: 4290, K20: 1200 ")

    print("\nStarting performance tests..\n")
    test_cpu()
    if torch.cuda.is_available():
        test_pci()
        test_cuda()
    else:
        print("\nWARNING: CUDA is not available.")
    print("\nPerformance Test finished.\n")
