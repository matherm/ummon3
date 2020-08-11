# -*- coding: utf-8 -*-
# @Author: Daniel
# @Date:   2020-07-30 09:11:06
# @Last Modified by:   Daniel
# @Last Modified time: 2020-08-10 15:41:04
import numpy as np
import torch
from ummon.utils.average_utils import *


class TestUtilsAverageUtils:

    def test_avg_single_value(self):
        data = np.arange(0, 20)
        avg_f = OnlineAverage()
        avg_f.reset()
        for v in data:
            avg = avg_f(v)
        assert np.isclose(avg, np.mean(data))

    def test_avg_single_value_torch(self):
        data = torch.arange(0, 20) + 0.1
        avg_f = OnlineAverage()
        avg_f.reset()
        for v in data:
            avg = avg_f(v)
        assert np.isclose(avg, torch.mean(data))

    def test_partly_avg_value_torch(self):
        data = torch.arange(0, 20).view(10, 2) + 0.1
        avg_f = OnlineAverage()
        avg_f.reset()
        for v in data:
            avg = avg_f(v)
        assert np.isclose(avg, torch.mean(data))

    def test_partly_avg_value_different_size(self):
        data = [[0, 1], [2, 3, 4], [5], [6, 7], [8, 9, 10],
                [11], [12], [13, 14, 15, 16, 17, 18, 19]]
        avg_f = OnlineAverage()
        avg_f.reset()
        for v in data:
            avg = avg_f(v)
        assert np.isclose(avg, np.mean(np.arange(0,20)))
