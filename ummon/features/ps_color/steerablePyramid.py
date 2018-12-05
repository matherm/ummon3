import sys
sys.path.insert(0, '../prog')

from .psUtils import *
import numpy as np
import scipy.special

import warnings


class SteerablePyramid:
    def __init__(self, img, height, order=3, twidth=1, hilbertTransformPairs=True):
        """

        :param img:
        :param height:
        :param order:
        :param twidth:
        :param hilbertTransformPairs:
        """

        if not isinstance(img, np.ndarray):
            img = np.array(img)
        assert img.ndim == 2, 'Input has to be two dimensional.'
        self._img = img

        max_height = np.floor(np.log2(min(self._img.shape)))
        assert height < max_height, 'Cannot build pyramid higher than {} levels.'.format(max_height)
        self._height = height

        if not 0 <= order <= 15:
            warnings.warn('order must be an integer within the range [0,15], value is being clipped.')
        elif not isinstance(order, int):
            warnings.warn('order has to be an integer, value is being converted.')
        self._nbands = int(np.clip(int(order), 0, 15) + 1)

        if twidth <= 0:
            # warnings.warn('twidth must be positive, value is being set to 1.')
            twidth = 1
        self._twidth = twidth

        self._htp = hilbertTransformPairs

    def buildSCFpyr(self):
        if self._nbands % 2:
            harmonics = np.arange(((self._nbands - 1) / 2) + 1) * 2
        else:
            harmonics = np.arange(self._nbands / 2) * 2 + 1

        angles = np.pi * np.arange(self._nbands) / self._nbands

        steeringMatrix = self.__computeSteeringMatrix(harmonics, angles, 'even')

        ny, nx = self._img.shape
        vx = np.arange(-np.ceil(nx / 2), nx // 2) / (nx / 2)
        vy = np.arange(-np.ceil(ny / 2), ny // 2) / (ny / 2)
        x, y = np.meshgrid(vx, vy)

        angle = np.arctan2(y, x)
        logRad = np.sqrt(x ** 2 + y ** 2)
        logRad[int(ny // 2), int(nx // 2)] = logRad[int(ny // 2), int((nx // 2) - 1)]
        logRad = np.log2(logRad)

        xRcos, yRcos = rCosFn(self._twidth, -self._twidth / 2)
        yRcos = np.sqrt(yRcos)

        yIRcos = np.sqrt(1. - (yRcos ** 2))
        lo0Mask = pointOp(logRad, yIRcos, xRcos)
        imgDFT = np.fft.fftshift(np.fft.fft2(self._img))
        lo0DFT = imgDFT * lo0Mask

        pyr, pind = self.__buildSCFpyrLevs(lo0DFT, logRad, xRcos, yRcos, angle, self._height)

        hi0Mask = pointOp(logRad, yRcos, xRcos)
        hi0DFT = imgDFT * hi0Mask
        hi0 = np.fft.ifft2(np.fft.ifftshift(hi0DFT))

        pyr = np.r_[hi0.real.ravel(order='F'), pyr.ravel()]
        pind = np.r_[np.array([hi0.shape]), pind.reshape((1, -1)) if pind.ndim == 1 else pind]

        return pyr, pind, steeringMatrix, harmonics

    @staticmethod
    def __computeSteeringMatrix(harmonics, angles, evenOrOdd='even'):
        assert evenOrOdd == 'even' or evenOrOdd == 'odd', 'evenOrOdd has to be either even or odd'

        numh = 2 * len(harmonics) - int(np.any(harmonics == 0))
        invMat = np.zeros((len(angles), numh))
        col = 0
        for h in harmonics:
            x = h * angles
            if h == 0:
                invMat[:, col] = np.ones((len(angles)))
                col += 1
            elif evenOrOdd == 'odd':
                invMat[:, col] = np.sin(x)
                invMat[:, col + 1] = -np.cos(x)
                col += 2
            else:
                invMat[:, col] = np.cos(x)
                invMat[:, col + 1] = np.sin(x)
                col += 2

        r = np.linalg.matrix_rank(invMat)
        if r != numh and r != len(angles):
            warnings.warn('inverse matrix is not full rank')

        return np.linalg.pinv(invMat)

    def __buildSCFpyrLevs(self, loDFT, logRad, xRcos, yRcos, angle, height):
        if height <= 0:
            lo0 = np.fft.ifft2(np.fft.ifftshift(loDFT))
            pyr = lo0.real.T.ravel()
            pind = np.array(lo0.shape)
        else:
            bands = np.zeros((np.prod(np.array(loDFT.shape)), self._nbands)).astype(np.complex)
            bind = np.zeros((self._nbands, 2))

            xRcos = xRcos - np.log2(2)
            lutSize = 1024
            xCosN = np.pi * np.arange(-(2 * lutSize + 1), (lutSize + 2)) / lutSize
            order = self._nbands - 1

            const = (2 ** (2 * order)) * (scipy.special.factorial(order) ** 2) / \
                    (self._nbands * scipy.special.factorial(2 * order))

            if self._htp:
                alpha = ((np.pi + xCosN) % (2 * np.pi)) - np.pi
                yCosN = 2 * np.sqrt(const) * (np.cos(xCosN) ** order) * (np.abs(alpha) < (np.pi / 2)).astype(int)
            else:
                yCosN = np.sqrt(const) * (np.cos(xCosN) ** order)

            hiMask = pointOp(logRad, yRcos, xRcos)

            for b in range(self._nbands):
                angleMask = pointOp(angle, yCosN, xCosN + (np.pi * b / self._nbands))
                if self._htp:
                    bandDFT = (np.complex(0, -1) ** order) * loDFT * angleMask * hiMask
                else:
                    bandDFT = (np.complex(0, 1) ** order) * loDFT * angleMask * hiMask
                band = np.fft.ifft2(np.fft.ifftshift(bandDFT))
                if self._htp:
                    bands[:, b] = band.ravel(order='F')
                else:
                    bands[:, b] = band.real.ravel(order='F')
                bind[b] = np.array(band.shape)

            dims = np.array(loDFT.shape)
            loDims = np.ceil((dims - .5) / 2)
            loCtr = np.ceil((loDims + .5) / 2)
            loStart = (np.ceil((dims + .5) / 2) - loCtr).astype(int)
            loEnd = (loStart + loDims).astype(int)

            logRad = logRad[loStart[0]: loEnd[0], loStart[1]: loEnd[1]]
            angle = angle[loStart[0]: loEnd[0], loStart[1]: loEnd[1]]
            loDFT = loDFT[loStart[0]: loEnd[0], loStart[1]: loEnd[1]]
            yIRcos = np.abs(np.sqrt(1. - yRcos ** 2))
            loMask = pointOp(logRad, yIRcos, xRcos)

            loDFT = loMask * loDFT

            nPyr, nPind = self.__buildSCFpyrLevs(loDFT, logRad, xRcos, yRcos, angle, height - 1)

            pyr = np.r_[bands.ravel(order='F'), nPyr.ravel(order='F')]
            pind = np.r_[bind, np.array([nPind]) if nPind.ndim == 1 else nPind]

        return pyr.reshape((-1, 1)), pind
