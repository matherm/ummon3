# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:44:12 2018

Originally taken from: https://github.com/leongatys/PytorchNeuralStyleTransfer

@author: Fabian Freiberg
@Last Modified by:   Daniel Dold
@Last Modified time: 2019-04-11 12:47:07
"""
import torch.nn as nn
import torch

__all__ = ['GramMatrix', 'GramDiag']


class GramMatrix(nn.Module):
    ##
    # @brief      { function_description }
    ##
    # @param      self  The object
    # @param      x     input Tensor shape [batch x in_feature x height x width]
    ##
    # @return     gram matrix shape [batch x in_feature x in_feature]
    ##
    def forward(self, x):
        b, c, h, w = x.size()
        F = x.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w)
        return G


class GramDiag(nn.Module):
    """
    docstring for GramDiag
    """

    ##
    # @brief      Constructs the object. #
    #
    # @param      self                   The object
    # @param      gram_diagonal_squared  The gram diagonal squared (means
    #                                    vector^4) #
    #
    def __init__(self, gram_diagonal_squared=False):
        super().__init__()
        self.__gram_diagonal_squared = gram_diagonal_squared

    ##
    # @brief      calculate gram diagonal
    ##
    # @param      self  The object
    # @param      x     input tensor shape [batch x in_feature x height x width]
    ##
    # @return     gram diagonal shape [batch x in_feature]
    ##
    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, 1, h * w)
        gram_diag = None
        for b in range(x.size(0)):
            if self.__gram_diagonal_squared:
                z = torch.bmm(x[b] * x[b], (x[b] * x[b]).transpose(2, 1))
            else:
                z = torch.bmm(x[b], x[b].transpose(2, 1))
            if isinstance(gram_diag, torch.Tensor):
                gram_diag = torch.cat(gram_diag, z)
            else:
                gram_diag = z
        gram_diag = torch.squeeze(gram_diag).unsqueeze(0)
        return gram_diag.div_(h * w)


def test():
    in_t = torch.linspace(-2, 2, 200).view(1, 8, 5, 5)
    gram_m = GramMatrix()(in_t)
    gram_m_d = torch.diagonal(gram_m, dim1=1, dim2=2)

    gram_d = GramDiag()(in_t)
    print("calculation difference: ", gram_d.data - gram_m_d.data)
    assert torch.all(1e-6 > torch.abs(gram_d.data -
                                      gram_m_d.data)), "gram diagonal from gram matrix and gram_diagonal calculation differs"


if __name__ == '__main__':
    test()
