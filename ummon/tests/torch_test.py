import unittest
import torch
import numpy

class TestTorch(unittest.TestCase):

    def test_pytorch(self):
        A =  numpy.random.randn(100,100)
        B =  numpy.matmul(A, A)
        C =  torch.DoubleTensor(100, 100).zero_() + torch.from_numpy(A)
        D =  C.matmul(C).numpy()
        self.assertTrue(numpy.allclose(B, D))

if __name__ == '__main__':
    unittest.main()