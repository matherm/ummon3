from __future__ import division
import numpy as np

from .auxiliary import compare


class Params:
    """
    Prams class takes parameters as dictionary and allows some other operations.

    """

    def __init__(self, params):
        """

        Args:
            params: parameters as dictionary

        Returns:
            parameters as parameter object

        """
        self.params = {k: np.atleast_2d(v.copy()) for k,v in params.items()}

    def add(self, defect):
        """
        to add defect difference to texture params.

        """
        params_dic = {k: (v.copy() + defect.params[k].copy()) for k,v in self.params.items()}

        params = Params(params_dic)

        return params
        # ToDo: change order in defect.py: subtract original from diff to get defect difference


    def subtract(self, defect):
        """
        to subtract defect difference from defect params. Return params of texture without defect.

        """
        params_dic = {k: v.copy() - defect.params[k].copy() for k,v in self.params.items()}

        params = Params(params_dic)

        return params


    def average(self, params2):

        params_dic = {k: np.array(v.copy() + params2.params[k].copy())/2 for k,v in self.params.items()}

        params = Params(params_dic)

        return params


    def get(self):

        return self.params

    def get_dic(self):

        params_dic = {k: v.copy() for k,v in self.params.items()}

        # for 1-dimensional statistics - cut off last dimension again
        for key in self.params:
            if key == 'magMeans' or key == 'pixelStats' or key == 'varianceHPR':
                params_dic[key] = np.squeeze(self.params[key], axis=0)

                if key == 'varianceHPR':
                    params_dic[key] = self.params[key][0]

        return params_dic

    def compare(self, params2, reltol=0, abstol=1.e-1):

        params_dic = self.get_dic()
        params2_dic = params2.get_dic()
        dic = dict(measure='percentual_close',rtol=reltol, atol=abstol)
        return {k:compare(params_dic[k],params2_dic[k],k, **dic) for k in params_dic}