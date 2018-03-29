import unittest
import torch
import numpy

class TestCuda(unittest.TestCase):

    def test_cuda(self):
        self.assertTrue(torch.cuda.is_available())
        if torch.cuda.is_available():
            CPU = torch.IntTensor(1000, 1000).zero_()
            torch.cuda.synchronize()
            for i in range(2):
                GPU = CPU.cuda()
                CPU = GPU.cpu()
            torch.cuda.synchronize()
            self.assertTrue(numpy.allclose(CPU.numpy(),GPU.cpu().numpy()))

if __name__ == '__main__':
    unittest.main()
