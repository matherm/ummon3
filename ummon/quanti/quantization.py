# -*- coding: utf-8 -*-
# @Author: daniel
# @Date:   2018-12-21 16:09:20
# @Last Modified by:   daniel
# @Last Modified time: 2019-04-12 18:51:39
from collections import UserDict
import copy
import numpy as np
import torch
import logging
from .quantization_helpers import quantize, distribution_shifting, clip, simulate_int_overflow, clip_r

__log = logging.getLogger('quanti_log')


class stdParam(UserDict):
    """
    A dictionary containing all quantization parameters with attribute-style
    access. It maps attribute access to the real dictionary.
    """
    my_keys = ('bitwith', 'mean', 'std', 'divisor',
               'handle_overflow', 'rounding', 'clip_range')

    ##
    # @brief      Constructs the object.
    ##
    # @param      self             The object
    # @param      bitwith          The simulated bitwith
    # @param      mean             The mean of models weigts
    # @param      std              The standard derivation of models weights
    # @param      divisor          The divisor acts as an hyperparameter for
    # weights transformation
    # @param      handle_overflow  Apply a layer wise integer overflow (if
    # False saturation is simulated)
    # @param      rounding         Use floor or round for values between
    # integer representation
    # @param      clip_range       The clipping range (default: {min:-1,
    # max:1})
    ##
    def __init__(self, bitwith, mean=0, std=1, divisor=1, handle_overflow=False, rounding=True, clip_range=dict(min=-1, max=1)):
        stdParam.__setattr__ = object.__setattr__
        stdParam.__getattr__ = object.__getattribute__
        my_dict = {'bitwith': bitwith,
                   'std': std,
                   'mean': mean,
                   'divisor': divisor,
                   'handle_overflow': handle_overflow,
                   'rounding': rounding,
                   'clip_range': clip_range
                   }
        super(stdParam, self).__init__(my_dict)
        """ Syntax candy """
        stdParam.__setattr__ = stdParam.__setitem__
        stdParam.__getattr__ = UserDict.__getitem__

    def __setitem__(self, key, value):
        if key in stdParam.my_keys:
            return super(stdParam, self).__setitem__(key, value)
        else:
            return value

    def __getstate__(self):
        ''' Needed for cPickle in .copy() '''
        return copy.copy(self.__dict__)

    def __setstate__(self, dict):
        ''' Needed for cPickle in .copy() '''
        self.__dict__.update(dict)

    def __deepcopy__(self, memodict={}):
        return stdParam(**(dict(self)))


def mean_std(parameters_iter):
    params = []
    for param in parameters_iter:
        params.extend(param.data.detach().cpu().numpy().reshape(-1))
    std = np.std(np.array(params))
    mean = np.mean(np.array(params))
    __log.debug(
        "standard deviation= {0}, mean= {1} of all weights and bias".format(std, mean))
    return mean.item(), std.item()


##
# @brief      Custom quantization function. Assuming a distribution shifting, a
#             quantization and clipping using the clipping range.
##
class quantizationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, param: stdParam):
        # print("forward")
        if param.mean is 0 and param.std is 1 and param.divisor is 1:
            out = input
        else:
            out = distribution_shifting(
                input, param.mean, param.std, param.divisor)
        out = quantize(out, param.bitwith, param.rounding)
        if param.handle_overflow:
            out = simulate_int_overflow(out)
        out = clip_r(out, **param.clip_range)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # print("calc_grad")
        return grad_output, None


##
# @brief      Wrapped the quantization function in a pytorch module 
#              if mean == 0 and std == 1 and divisor == 1 =>
#                  distribution shifting will not be applied.
#              Otherwise the following operation will be performed
#                  - distribution shifting
#                  - quantization
#                  - clipping
##
class Quantization(torch.nn.Module):

    ##
    ## @brief      Constructs the object.
    ##
    ## @param      self   The object
    ## @param      param  the parameters of type stdParam which contains all
    ##                    necessary quantization parameters
    ##
    def __init__(self, param: stdParam):
        super(Quantization, self).__init__()
        self.param = param

    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return quantizationFunction.apply(input, self.param)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'param={}'.format(
            self.param
        )


def test():
    param = stdParam(bitwith=3)
    t = torch.tensor([0, 0.2, 0.3, 0.5, -1, 2])
    t.requires_grad = True
    out = quantizationFunction.apply(t, param)
    out_expected = torch.tensor([0.0, 0.25, 0.25, 0.5, -0.75, 0.75])
    print("In:  {} => \nOut: {} \nExpected:{}".format(t, out, out_expected))
    assert torch.equal(out, out_expected),\
        "error at quantization in forwardpath "

    out = out.sum()
    out.backward()
    # print("Grad: ", t.grad)
    grad_expected = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert torch.equal(t.grad, grad_expected),\
        "error at gradient calculation while quantization was involved "


if __name__ == '__main__':
    test()
