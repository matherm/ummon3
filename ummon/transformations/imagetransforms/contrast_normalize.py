import torch
import numpy as np

class ContrastNormalize():
    """
    Contrast normalizes a tensor image with per channel. 
    """
    
    def __call__(self, x):
        assert torch.is_tensor(x)
        assert x.dim() == 3
        if x.size(0) == 1 or x.size(0) == 3:
            x = x / torch.norm(x.view(x.size(0), -1), dim=1).view(x.size(0), 1, 1)
        else:
            raise ValueError("Shape not understood. Was", x.size())
        return x