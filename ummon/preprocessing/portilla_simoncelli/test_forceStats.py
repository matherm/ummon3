from __future__ import division
import numpy as np

import impy as ip
from scipy.io import loadmat
from scipy.io import savemat

from .auxiliary import forceAutoCorr, adjustCorr1s, \
        adjustCorr2s, signaltonoise, modskew, modkurt, mkAngle, \
        loadFromMat, compare, expand


class Test:
    def __init__(self):
        """

        Returns:

        """
        self.test = 0

    def test_forceAutoCorr(self, img, autoCorro, p):

        #matlab_result = loadFromMat('fautoCorr')
        matlab_result = loadmat('../matlab/textureSynth/fautoCorr.mat')['fautoCorr']
        result, snrV, Chf = forceAutoCorr(img, autoCorro, p)
        compare(result.real, matlab_result,'fautoCorr')
        path_name = "result/diff/fautoCorr.png"
        ip.save_img(result.real.copy(), path_name)
        path_name = "result/diff/fautoCorrMAT.png"
        ip.save_img(matlab_result.real.copy(), path_name)



    def test_adjustCorr1s(self, img, Co, mode, p):

        #matlab_result = loadFromMat('Corr1s')
        matlab_result = loadmat('../matlab/textureSynth/Corr1s.mat')['Corr1s']
        result, snr1, M = adjustCorr1s(img, Co, mode, p)
        compare(result, matlab_result,'Corr1s')
        path_name = "result/diff/Corr1s.png"
        ip.save_img(np.reshape(result[:,0].real.copy(), (128,128)), path_name)
        path_name = "result/diff/Corr1sMAT.png"
        ip.save_img(np.reshape(matlab_result[:,0].real.copy(), (128,128)), path_name)

    def test_adjustCorr2s(self, X, Cx, Y, Cxy, mode, p):

        #matlab_result = loadFromMat('Corr2s')
        matlab_result = loadmat('../matlab/textureSynth/Corr2s.mat')['Corr2s']
        result, snr1, snr2, Mx, My = adjustCorr2s(X, Cx, Y, Cxy, mode, p)
        compare(result,matlab_result,'Corr2s')
        path_name = "result/diff/Corr2s.png"
        ip.save_img(np.reshape(result[:,0].real.copy(), (128,128)), path_name)
        path_name = "result/diff/Corr2sMAT.png"
        ip.save_img(np.reshape(matlab_result[:,0].real.copy(), (128,128)), path_name)

    def test_signaltonoise(self, s, n):

        #matlab_result = loadFromMat('signaltonoise')
        matlab_result = loadmat('../matlab/textureSynth/signaltonoise.mat')['signaltonoise']
        result = signaltonoise(s, n)
        compare(result,matlab_result,'signaltonoise')


    def test_modskew(self, ch, sk, p):

        #matlab_result = loadFromMat('skew')
        matlab_result = loadmat('../matlab/textureSynth/skew.mat')['skew']
        result, snrk = modskew(ch, sk, p)
        compare(result, matlab_result,'skew')

        path_name = "result/diff/skew.png"
        ip.save_img(result.real.copy(), path_name)
        path_name = "result/diff/skewMAT.png"
        ip.save_img(matlab_result.real.copy(), path_name)



    def test_modkurt(self, ch, k, p):

        #matlab_result = loadFromMat('kurt')
        matlab_result = loadmat('../matlab/textureSynth/kurt.mat')['kurt']
        result, snrk = modkurt(ch, k, p)
        compare(result, matlab_result,'kurt')

        path_name = "result/diff/kurt.png"
        ip.save_img(result.real.copy(), path_name)
        path_name = "result/diff/kurtMAT.png"
        ip.save_img(matlab_result.real.copy(), path_name)



    def test_mkAngle(self, sz, phase, origin):

        #matlab_result = loadFromMat('angle')
        matlab_result = loadmat('../matlab/textureSynth/angle.mat')['angle']
        result = mkAngle(sz, phase, origin)
        compare(result, matlab_result,'angle')

        #print("result: ", result)
        #print("matlab_result: ", matlab_result)


    def test_expand(self, img, factor):

        #matlab_result = loadFromMat('expand')
        matlab_result = loadmat('../matlab/textureSynth/expand.mat')['expand']
        result = expand(img, factor)
        compare(result, matlab_result,'expand')
        path_name = "result/diff/expand.png"
        ip.save_img(result.real.copy(), path_name)
        path_name = "result/diff/expandMAT.png"
        ip.save_img(matlab_result.real.copy(), path_name)


