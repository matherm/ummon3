# -*- coding: utf-8 -*-
# @Author: Daniel
# @Date:   2019-04-16 15:12:51
# @Last Modified by:   Daniel
# @Last Modified time: 2019-04-16 15:15:16
import torch
from ummon.modules.gram import GramMatrix, GramDiag


def test():
    in_t = torch.linspace(-2, 2, 200).view(1, 8, 5, 5)
    gram_m = GramMatrix()(in_t)
    gram_m_d = torch.diagonal(gram_m, dim1=1, dim2=2)

    gram_d = GramDiag()(in_t)
    print("calculation difference: ", gram_d.data - gram_m_d.data)
    assert torch.all(1e-6 > torch.abs(gram_d.data -
                                      gram_m_d.data)), "gram diagonal from gram matrix and gram_diagonal calculation differs"

