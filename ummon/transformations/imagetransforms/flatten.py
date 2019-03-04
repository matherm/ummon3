import torch
import numpy as np

class FlattenTransform():

    def __call__(self, tensor):
        return tensor.contiguous().view(-1).clone()