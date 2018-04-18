import unittest
import torch
import numpy
from ummon.logger import Logger

class TestCuda(unittest.TestCase):

    def test_cuda(self):
        if not torch.cuda.is_available():
            Logger().error("CUDA is not available on your system.")
        else:
            CPU = torch.IntTensor(1000, 1000).zero_()
            torch.cuda.synchronize()
            for i in range(2):
                GPU = CPU.cuda()
                CPU = GPU.cpu()
            torch.cuda.synchronize()
            self.assertTrue(numpy.allclose(CPU.numpy(),GPU.cpu().numpy()))

if __name__ == '__main__':
    unittest.main()
