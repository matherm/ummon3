import torch
import numpy as np

class Binarize():
    
    def __init__(self, double=False):
        self.double = double
    
    def __call__(self, x):
        '''
        Binarize Image
        '''
        m = torch.distributions.Uniform(0, 1)
        xb = m.sample(x.size())
        bin_image = (x > xb).float() * 1
        if self.double == True:
            return bin_image.double()
        else:
            return bin_image
