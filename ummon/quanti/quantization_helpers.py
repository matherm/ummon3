# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2018-12-27 19:07:10
# @Last Modified by:   daniel
# @Last Modified time: 2019-03-19 22:25:29
import torch
import numpy as np


def scale(bits):
    return 2.0 ** (bits - 1)


def clip(x, bits, symmetric=True):
    dist = 1 / scale(bits)
    _max = 1 - dist
    if symmetric:
        _min = -1 + dist
    else:
        _min = -1
    return torch.clamp(x, min=_min, max=_max)


def clip_r(x, **_range):
    return torch.clamp(x, **_range)


def simulate_int_overflow(x):
    return ((x + 1) % 2) - 1


def shift(x):
    if x != 0.0:
        return 2 ** (np.round(np.log2(x)))
    else:
        return 1  # when x is 0, means x/shift = 0 => therefor shift can be one


def quantize(x, nb_bits: int, rounding: bool=True):
    # uniform distance = 2^(1-bits) = 1/(2^(bits-1))
    _scale = scale(nb_bits)
    # is torch.round same behavior as the integer arithmetic does (FixPU)
    if rounding:
        x = torch.round(x * _scale) / _scale
    else:
        x = torch.floor(x * _scale) / _scale
    return x


# not used by wage
def distribution_shifting(x, mean, std, divisor=1.0):
    return (x - mean) / (std * divisor)
