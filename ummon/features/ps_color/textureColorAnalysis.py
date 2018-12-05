import sys
sys.path.insert(0, '../prog')

from .psUtils import *
from .steerablePyramid import SteerablePyramid
from .reconstructFromPyramid import ReconstructFromPyramid

import numpy as np
import scipy.stats

import warnings


class TextureColorAnalysis:

    def __init__(self, original_image, number_of_scales=4, number_of_orientations=4, spatial_neighborhood=7):
        """

        Analyze texture for application of Portilla-Simoncelli model/algorithm.

        :param original_image:          original image (uint8, true color)
        :param number_of_scales:        number of scales
        :param number_of_orientations:  number of orientations
        :param spatial_neighborhood:    spatial neighborhood considered (Na x Na)
        """
        if isinstance(original_image, np.ndarray):
            self._im0 = original_image
        else:
            self._im0 = np.array(original_image)

        assert self._im0.ndim == 3, 'The input image has to be a three dimensional array!'
        assert self._im0.shape[2] == 3, 'The input image has to be of size [H x W x C], with C = 3'

        if not isinstance(number_of_scales, int):
            warnings.warn('number_of_scales has to be of type int, converting value {} to {}'.format(
                number_of_scales, int(number_of_scales)))
        self._nsc = int(number_of_scales)

        if not isinstance(number_of_scales, int):
            warnings.warn('number_of_orientation has to be of type int, converting value {} to {}'.format(
                number_of_orientations, int(number_of_orientations)))
        self._nor = int(number_of_orientations)

        assert spatial_neighborhood % 2, 'spatial neighborhood has to be odd'
        if not isinstance(spatial_neighborhood, int):
            warnings.warn('number_of_scales has to be of type int, converting value {} to {}'.format(
                spatial_neighborhood, int(spatial_neighborhood)))
        self._na = int(spatial_neighborhood)

        self._ny, self._nx, self._nclr = self._im0.shape
        self._im0 = self._im0.astype(float)
        self._perChannel = lambda f, x: np.array([f(_) for _ in x.T])

        self._pyr0 = None
        self._pInd0 = None
        self._rPyr0 = None
        self._aPyr0 = None

        self.pixelStats = None
        self.pixelStatsPCA = None
        self.pixelLPStats = None
        self.autoCorrReal = np.full((self._na, self._na, self._nsc + 2, self._nclr), np.nan)
        self.autoCorrMag = np.full((self._na, self._na, self._nsc, self._nor, self._nclr), np.nan)
        self.magMeans = None
        self.cousinMagCorr = np.zeros((self._nclr * self._nor, self._nclr * self._nor, self._nsc))
        self.parentMagCorr = np.zeros((self._nclr * self._nor, self._nclr * self._nor, self._nsc - 1))
        self.cousinRealCorr = np.zeros((2 * self._nclr * self._nor, 2 * self._nclr * self._nor, self._nsc + 1))
        self.parentRealCorr = np.zeros((self._nclr * self._nor, 2 * self._nclr * self._nor, self._nsc))
        self.varianceHPR = np.zeros((self._nclr, 1))
        self.colorCorr = None

    @staticmethod
    def __kurt(img):
        return scipy.stats.kurtosis(img, axis=None, fisher=False)

    @staticmethod
    def __skew(img):
        return scipy.stats.skew(img, axis=None)

    def __computePixelStats(self, img):
        mn = self._perChannel(np.min, img)
        mx = self._perChannel(np.max, img)
        var = self._perChannel(np.var, img)
        skew = self._perChannel(self.__skew, img)
        kurt = self._perChannel(self.__kurt, img)

        return mn, mx, var, skew, kurt

    def __computeAutoCorrOfBands(self, D):
        # Computing central autoCorr of lowBand
        im = np.zeros((self._ny, self._nx, self._nclr))
        skew0p = np.zeros((self._nsc + 1, self._nclr))
        kurt0p = np.zeros((self._nsc + 1, self._nclr))
        la = (self._na - 1) // 2
        for c in range(self._nclr):
            nband = len(self._pInd0)
            ch = pyrBand(self._pyr0[:, c], self._pInd0, nband)
            nly, nlx = ch.shape
            mPyr, mpInd, _, _ = SteerablePyramid(ch.real, 0, 0, False).buildSCFpyr()
            im[: nly, : nlx, c] = pyrBand(mPyr, mpInd, 2)
            le = int(min(min(nly, nlx) / 2 - 1, la))
            cy, cx = int(nly / 2 + 1), int(nlx / 2 + 1)
            ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(im[: nly, : nlx, c])) ** 2).real) / np.prod(ch.shape)
            ac = ac[cy - le - 1: cy + le, cx - le - 1: cx + le]
            self.autoCorrReal[la - le: la + le + 1, la - le: la + le + 1, self._nsc, c] = ac.copy()
            varI = ac[le, le]

            if varI / D[c] > 1e-6:
                skew0p[self._nsc, c] = np.mean(im[: nly, : nlx, c] ** 3) / (varI ** 1.5)
                kurt0p[self._nsc, c] = np.mean(im[: nly, : nlx, c] ** 4) / (varI ** 2)
            else:
                skew0p[self._nsc, c] = 0
                kurt0p[self._nsc, c] = 3

        # Computing central autoCorr of each Mag band and the autoCorr of the combined (non-oriented) band
        for c in range(self._nclr):
            for scale in range(self._nsc - 1, -1, -1):
                for orientation in range(self._nor):
                    nband = (scale * self._nor) + orientation + 2
                    ch = pyrBand(self._aPyr0[:, c], self._pInd0, nband)
                    nly, nlx = ch.shape
                    le = min(min(nly, nlx) / 2 - 1, la)
                    cy, cx = int(nly / 2 + 1), int(nlx / 2 + 1)
                    ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(ch)) ** 2).real) / np.prod(ch.shape)
                    ac = ac[cy - le - 1: cy + le, cx - le - 1: cx + le]
                    self.autoCorrMag[la - le: la + le + 1, la - le: la + le + 1, scale, orientation, c] = ac.T.copy()

                # Combining orientation bands
                bandNums = np.arange(1, self._nor + 1) + scale * self._nor + 1
                indB = pyrBandIndices(self._pInd0, bandNums[0])
                indE = pyrBandIndices(self._pInd0, bandNums[self._nor - 1])
                bandInds = np.arange(indB[0], indE[len(indE) - 1] + 1)

                fakePind = np.r_[
                    self._pInd0[bandNums[0] - 1, :].reshape((1, -1)),
                    self._pInd0[bandNums[0] - 1: bandNums[self._nor - 1] + 1, :]
                ]
                fakePyr = np.r_[
                    np.zeros((np.prod(fakePind[0, :]).astype(int))),
                    self._rPyr0[bandInds, c],
                    np.zeros((np.prod(fakePind[len(fakePind) - 1, :]).astype(int)))
                ]

                ch = ReconstructFromPyramid(fakePyr, fakePind, 1).reconSFpyr()
                nly, nlx = ch.shape
                le = min(min(nly, nlx) / 2 - 1, la)
                cy, cx = int(nly / 2 + 1), int(nlx / 2 + 1)
                im[: nly, : nlx, c] = expand(im[: int(nly / 2), : int(nlx / 2), c], 2).real / 4
                im[: nly, : nlx, c] += ch
                ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(im[: nly, : nlx, c])) ** 2).real) / np.prod(
                    ch.shape)
                ac = ac[cy - le - 1: cy + le, cx - le - 1: cx + le]
                self.autoCorrReal[la - le: la + le + 1, la - le: la + le + 1, scale, c] = ac.copy()
                varI = ac[le, le]

                if varI / D[c] > 1e-6:
                    skew0p[scale, c] = np.mean(im[: nly, : nlx, c] ** 3) / (varI ** 1.5)
                    kurt0p[scale, c] = np.mean(im[: nly, : nlx, c] ** 4) / (varI ** 2)
                else:
                    skew0p[scale, c] = 0
                    kurt0p[scale, c] = 3

        self.pixelLPStats = np.r_[skew0p, kurt0p]

    def __computeAutoCorrPCABands(self, imPCA):
        # Central autoCorr of the PCA bands
        la = (self._na - 1) // 2
        for c in range(self._nclr):
            ac = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(imPCA[:, :, c])) ** 2).real) / (self._ny * self._nx)
            cy, cx = int(self._ny / 2 + 1), int(self._nx / 2 + 1)
            ac = ac[cy - la - 1: cy + la, cx - la - 1: cx + la]
            self.autoCorrReal[la - la: la + la + 1, la - la: la + la + 1, self._nsc + 1, c] = ac.copy()

    def __subtractMeanOfMagnitude(self):
        # Subtracting mean of magnitude
        self.magMeans = np.zeros((len(self._pInd0), self._nclr))
        for c in range(self._nclr):
            for i in range(len(self._pInd0)):
                ind = pyrBandIndices(self._pInd0, i + 1)
                self.magMeans[i, c] = np.mean(self._aPyr0[ind, c])
                self._aPyr0[ind, c] -= self.magMeans[i, c]

    def __subtractMeanOfLowBand(self):
        # Subtracting mean of lowBand
        nband = len(self._pInd0)
        for c in range(self._nclr):
            self._pyr0[pyrBandIndices(self._pInd0, nband), c] = \
                self._pyr0[pyrBandIndices(self._pInd0, nband), c].real - np.mean(
                    self._pyr0[pyrBandIndices(self._pInd0, nband), c].real)
        self._rPyr0 = self._pyr0.real
        self._aPyr0 = np.abs(self._pyr0)

    def __buildSteerablePyramid(self, imPCA):
        # Building steerable pyramid
        for c in range(self._nclr):
            pyr, self._pInd0, _, _ = SteerablePyramid(imPCA[:, :, c], self._nsc, self._nor - 1, True).buildSCFpyr()
            if c == 0:
                self._pyr0 = pyr
            else:
                self._pyr0 = np.c_[self._pyr0, pyr]
        assert not np.any(self._pInd0 % 2), 'Algorithm will fail: Some bands have odd dimensions!'

    def __computeCrossCorrs(self):
        # Computing the cross-correlation matrices of the coefficient magnitudes pyramid at the different levels and
        # orientations
        for scale in range(self._nsc):
            firstBnum = scale * self._nor + 2
            cousinSz = int(np.prod(self._pInd0[firstBnum]))
            ind = pyrBandIndices(self._pInd0, firstBnum)
            cousinInd = ind[0] + np.arange(self._nor * cousinSz)

            cousins = np.zeros((cousinSz, self._nor, self._nclr))
            rCousins = np.zeros((cousinSz, self._nor, self._nclr))
            parents = np.zeros((cousinSz, self._nor, self._nclr))
            rParents = np.zeros((cousinSz, 2 * self._nor, self._nclr))

            for c in range(self._nclr):
                if scale < self._nsc - 1:
                    for orientation in range(self._nor):
                        nband = (scale + 1) * self._nor + orientation + 2

                        pyrBandExp = expand(pyrBand(self._pyr0[:, c], self._pInd0, nband), 2) / 4
                        pyrBandExp = np.sqrt(pyrBandExp.real ** 2 + pyrBandExp.imag ** 2) * np.exp(
                            2 * np.complex(0, 1) * np.arctan2(pyrBandExp.real, pyrBandExp.imag))
                        rParents[:, orientation, c] = pyrBandExp.real.ravel()
                        rParents[:, self._nor + orientation, c] = pyrBandExp.imag.ravel()

                        pyrBandExp = np.abs(pyrBandExp)
                        parents[:, orientation, c] = (pyrBandExp.T - np.mean(pyrBandExp)).ravel(order='F')
                else:
                    pyrBandExp = expand(pyrLow(self._pyr0[:, c], self._pInd0), 2).T.real / 4
                    rParents[:, : 5, c] = np.r_[
                        pyrBandExp.ravel(order='F'),
                        np.roll(pyrBandExp, 2, axis=1).ravel(order='F'),
                        np.roll(pyrBandExp, -2, axis=1).ravel(order='F'),
                        np.roll(pyrBandExp, 2, axis=0).ravel(order='F'),
                        np.roll(pyrBandExp, -2, axis=0).ravel(order='F')
                    ].reshape(rParents[:, : 5, c].shape, order='F')
                    parents = np.array([])

                cousins[:, :, c] = np.reshape(self._aPyr0[cousinInd, c], (cousinSz, self._nor), order='F')
                rCousins[:, :, c] = np.reshape(self._pyr0[cousinInd, c].real, (cousinSz, self._nor), order='F')

            nCousins = cousins.shape[1] * self._nclr
            if parents.ndim == 1:
                nParents = 0
            else:
                nParents = parents.shape[1] * self._nclr
            cousins = cousins.reshape((cousinSz, nCousins), order='F')
            parents = parents.reshape((cousinSz, nParents), order='F')

            self.cousinMagCorr[:, :, scale] = (cousins.T @ cousins) / cousinSz

            if nParents > 0:
                self.parentMagCorr[:, :, scale] = (cousins.T @ parents) / cousinSz

            if scale == self._nsc - 1:
                rParents = rParents[:, : 5, :]

            nrParents = rParents.shape[1] * self._nclr
            nrCousins = rCousins.shape[1] * self._nclr
            rCousins = rCousins.reshape((cousinSz, nrCousins), order='F')
            rParents = rParents.reshape((cousinSz, nrParents), order='F')

            self.cousinRealCorr[: nrCousins, : nrCousins, scale] = (rCousins.T @ rCousins) / cousinSz
            if nrParents > 0:
                self.parentRealCorr[: nrCousins, : nrParents, scale] = (rCousins.T @ rParents) / cousinSz

                if scale == self._nsc - 1:
                    self.cousinRealCorr[: nrParents, : nrParents:, self._nsc] = (rParents.T @ rParents) / cousinSz

    def __computeVHPR(self):
        # Calculating the variance of the HF residual.
        for c in range(self._nclr):
            channel = self._pyr0[pyrBandIndices(self._pInd0, 1), c].real
            self.varianceHPR[c] = np.mean(channel ** 2)

    def analyze(self):
        """

        :return:
        """
        nth = np.log2(min(self._ny, self._nx) / self._na)
        if nth < self._nsc:
            warnings.warn('_na will be cut off for levels above {}'.format(int(nth + 1)))

        imPCA = np.reshape(self._im0, (self._ny * self._nx, self._nclr))
        mean0 = np.mean(imPCA, axis=0)
        imPCA = imPCA - mean0
        self.colorCorr = np.array((imPCA.T @ imPCA) / (self._ny * self._nx))
        D, V = np.linalg.eig(self.colorCorr)
        imPCA = imPCA @ V @ np.linalg.pinv(np.diag(np.sqrt(D)))
        imPCA = np.reshape(imPCA, (self._ny, self._nx, self._nclr))
        imPCA = imPCA[:, :, [1, 2, 0]] * [1, -1, 1]

        mn0, mx0, var0, skew0, kurt0 = self.__computePixelStats(self._im0)
        self.pixelStats = np.array([mean0, var0, skew0, kurt0, mn0, mx0])

        mnPCA, mxPCA, varPCA, skewPCA, kurtPCA = self.__computePixelStats(imPCA)
        self.pixelStatsPCA = np.array([skewPCA, kurtPCA, mnPCA, mxPCA])

        self.__buildSteerablePyramid(imPCA)
        self.__subtractMeanOfLowBand()
        self.__subtractMeanOfMagnitude()
        self.__computeAutoCorrPCABands(imPCA)
        self.__computeAutoCorrOfBands(D)
        self.__computeCrossCorrs()
        self.__computeVHPR()

    def getFeaturesList(self):
        """

        :return:
        """
        return [
            self.pixelStats,
            self.pixelStatsPCA,
            self.pixelLPStats,
            self.autoCorrReal,
            self.autoCorrMag,
            self.magMeans,
            self.cousinMagCorr,
            self.parentMagCorr,
            self.cousinRealCorr,
            self.parentRealCorr,
            self.varianceHPR,
            self.colorCorr
        ]

    def getFeaturesArray(self):
        return np.r_[
            self.pixelStats.ravel(),
            self.pixelStatsPCA.ravel(),
            self.pixelLPStats.ravel(),
            self.autoCorrReal.ravel(),
            self.autoCorrMag.ravel(),
            self.magMeans.ravel(),
            self.cousinMagCorr.ravel(),
            self.parentMagCorr.ravel(),
            self.cousinRealCorr.ravel(),
            self.parentRealCorr.ravel(),
            self.varianceHPR.ravel(),
            self.colorCorr.ravel()
        ]

    def getFeaturesDict(self):
        return {
            'pixelStats': self.pixelStats,
            'pixelStatsPCA': self.pixelStatsPCA,
            'pixelLPStats': self.pixelLPStats,
            'autoCorrReal': self.autoCorrReal,
            'autoCorrMag': self.autoCorrMag,
            'magMeans': self.magMeans,
            'cousinMagCorr': self.cousinMagCorr,
            'parentMagCorr': self.parentMagCorr,
            'cousinRealCorr': self.cousinRealCorr,
            'parentRealCorr': self.parentRealCorr,
            'varianceHPR': self.varianceHPR,
            'colorCorr': self.colorCorr
        }
