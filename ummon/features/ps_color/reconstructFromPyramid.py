import sys
sys.path.insert(0, '../prog')

from .psUtils import *
import numpy as np
import scipy.special


class ReconstructFromPyramid:
    def __init__(self, pyr, indices, levels=-1, bands=-1, twidth=1):
        self._pyr = pyr
        if not isinstance(indices, np.ndarray):
            self._pInd = np.array(indices).astype(int)
        else:
            self._pInd = indices.astype(int)

        if not isinstance(levels, np.ndarray):
            self._levels = np.array(levels).astype(int)
        else:
            self._levels = levels.astype(int)

        if not isinstance(bands, np.ndarray):
            self._bands = np.array(bands).astype(int)
        else:
            self._bands = bands.astype(int)

        self._twidth = twidth
        self._nbands = self.__sPyrNumBands()
        self._height = self.__sPyrHt()

    def reconSFpyr(self):
        maxLev = self._height + 1
        if self._levels == -1:
            self._levels = np.arange(maxLev)
        else:
            assert np.any(self._levels <= maxLev) or np.any(self._levels >= 0), \
                'level numbers must be in the range [0, {}]'.format(maxLev)

        if self._bands == -1:
            self._bands = np.arange(self._nbands)
        else:
            assert np.any(self._bands < self._nbands) or np.any(self._bands >= 0), \
                'band numbers must be in the range [0, {}]'.format(self._nbands)

        ny, nx = self._pInd[0]
        vx = np.arange(-np.ceil(nx / 2), nx // 2) / (nx / 2)
        vy = np.arange(-np.ceil(ny / 2), ny // 2) / (ny / 2)
        x, y = np.meshgrid(vx, vy)

        angle = np.arctan2(y, x)
        logRad = np.sqrt(x ** 2 + y ** 2)
        logRad[int(ny // 2), int(nx // 2)] = logRad[int(ny // 2), int((nx // 2) - 1)]
        logRad = np.log2(logRad)

        xRcos, yRcos = rCosFn(self._twidth, -self._twidth / 2)
        yRcos = np.sqrt(yRcos)

        yIRcos = np.sqrt(np.abs(1. - (yRcos ** 2)))

        if len(self._pInd) == 2:
            if np.any(self._levels == 1):
                resDFT = np.fft.fftshift(np.fft.fft2(pyrBand(self._pyr, self._pInd, 2)))
            else:
                resDFT = np.zeros((self._pInd[1]))
        else:
            resDFT = self.__reconSFpyrLevs(self._pyr[np.prod(self._pInd[0]): len(self._pyr)],
                                           self._pInd[1: len(self._pInd)], logRad, xRcos, yRcos, angle, self._levels)

        lo0Mask = pointOp(logRad, yIRcos, xRcos)
        resDFT *= lo0Mask

        if np.any(self._levels == 0):
            hi0Mask = pointOp(logRad, yRcos, xRcos)
            subMtx = np.reshape(
                np.array([self._pyr[:, i] for i in self._pyr.shape[1]]).flatten()[:np.prod(self._pInd[0])],
                (self._pInd[0]))
            hiDFT = np.fft.fftshift(np.fft.fft2(subMtx))
            resDFT += hiDFT * hi0Mask

        return np.fft.ifft2(np.fft.ifftshift(resDFT)).real

    def __reconSFpyrLevs(self, pyr, pInd, logRad, xRcos, yRcos, angle, levels):
        loInd = self._nbands
        dims = pInd[0]

        xRcos = xRcos - np.log2(2)

        if np.any(levels > 1):
            loDims = np.ceil((dims - .5) / 2)
            loCtr = np.ceil((loDims + .5) / 2)
            loStart = (np.ceil((dims + .5) / 2) - loCtr).astype(int)
            loEnd = (loStart + loDims).astype(int)
            nLogRad = logRad[loStart[0]: loEnd[0], loStart[1]:loEnd[1]]
            nAngle = angle[loStart[0]: loEnd[0], loStart[1]:loEnd[1]]

            if len(self._pInd) > loInd:
                nResDFT = self.__reconSFpyrLevs(pyr[np.sum(np.prod(pInd[:loInd - 1])): len(self._pyr)],
                                                pInd[loInd: len(pInd) - 1], nLogRad, xRcos, yRcos, nAngle, levels - 1)
            else:
                nResDFT = np.fft.fftshift(np.fft.fft2(pyrBand(pyr, pInd, loInd)))

            yIRcos = np.sqrt(np.abs(1. - (yRcos ** 2)))
            loMask = pointOp(nLogRad, yIRcos, xRcos)

            resDFT = np.zeros(dims).astype(np.complex)
            resDFT[loStart[0]: loEnd[0], loStart[1]: loEnd[1]] = nResDFT * loMask

        else:
            resDFT = np.zeros(dims).astype(np.complex)

        if np.any(levels == 1):
            lutSize = 1024
            xCosN = np.pi * np.arange(-(2 * lutSize + 1), (lutSize + 2)) / lutSize
            order = self._nbands - 1

            const = (2 ** (2 * order)) * (scipy.special.factorial(order) ** 2) / \
                    (self._nbands * scipy.special.factorial(2 * order))
            yCosN = np.sqrt(const) * (np.cos(xCosN) ** order)
            hiMask = pointOp(logRad, yRcos, xRcos)

            ind = 0
            for b in range(self._nbands):
                if np.any(self._bands == b):
                    angleMask = pointOp(angle, yCosN, xCosN + (np.pi * b / self._nbands))
                    band = np.reshape(pyr[ind: ind + np.prod(dims)], dims, order='F')
                    bandDFT = np.fft.fftshift(np.fft.fft2(band))
                    resDFT += (np.complex(0, 1) ** order) * bandDFT * angleMask * hiMask
                ind += np.prod(dims)

        return resDFT

    def __sPyrNumBands(self):
        if len(self._pInd) == 2:
            nbands = 0
        else:
            b = 3
            while b <= (len(self._pInd) - 1) and np.all(self._pInd[b - 1, :] == self._pInd[1, :]):
                b += 1
            nbands = b - 2
        return nbands

    def __sPyrHt(self):
        if len(self._pInd) > 2:
            ht = (len(self._pInd) - 2) / self._nbands
        else:
            ht = 0
        return ht
