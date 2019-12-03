import torch
import numpy as np

class Normalize():
    """
    Normalize a tensor image with mean and standard deviation per channel. 
    """

    def __init__(self, with_mean=True, with_std=True):

        self.with_mean = with_mean
        self.with_std = with_std  


    def __call__(self, x):
        assert torch.is_tensor(x)
        assert x.dim() == 3
        if x.size(0) == 1 or x.size(0) == 3:
            if self.with_mean:
                x = x - x.view(x.size(0), -1).mean(1).view(x.size(0), 1, 1) # (C, W, H)
            if self.with_std:
                x = x / x.view(x.size(0), -1).std(1).view(x.size(0), 1, 1)  # (C, W, H)
        elif x.size(2) == 1 or x.size(2) == 3:
            if self.with_mean:
                x = x - x.view(-1, x.size(2)).mean(0).view(1, 1, x.size(2)) # (C, W, H)
            if self.with_std:
                x = x / x.view(-1, x.size(2)).std(0).view(1,1, x.size(2))  # (C, W, H)
        else:
            raise ValueError("Shape not understood. Was", x.size())
        return x