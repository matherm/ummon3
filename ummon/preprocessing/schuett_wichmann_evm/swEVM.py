#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Michael Grunwald at 09.08.2018
"""

import numpy as np
from scipy.signal import convolve

__all__ = ['swEMV']


class swEMV:
    """
        Schuett Wichmann Early Vision Model (EVM)

        Schuett, H. H., & Wichmann, F. A. (2017). An image-computable psychophysical spatial vision model. Journal of Vision, 17(12):12, 1–35, doi:10.1167/17.12.12.

        """

    def __init__(self):
        """
        Parameters
        ----------

        """
        pass

    def filter_logGabor(self, imSize, degSize, freq, orientation, bw, f, orient0=None):

        """
        This function creates a log-Gabor filter in frequency space.
        It takes an image size in px and degrees, a frequency in cyc/deg an orientation(0-2pi) and a bandwidth [freq,orientation] as input.
        """

        if orient0.any() == None:
            x = np.zeros((1, imSize[1]))
            y = np.zeros((1, imSize[0]))
            x[0] = np.arange(0, imSize[1])
            y[0] = np.arange(0, imSize[0])
            x = (x - np.ceil(np.mean(x))) / degSize[1]
            y = (y - np.ceil(np.mean(y))) / degSize[0]
            f = np.sqrt((x ** 2) + ((y ** 2).T))
            orient0 = np.arctan2(x, y.T).T

        filter = np.zeros((imSize[0], imSize[1], np.size(orientation)))

        for iOrient in range(0, np.size(orientation)):
            orient = orient0 - orientation[iOrient] + np.pi
            orient = np.mod(orient, 2 * np.pi) - np.pi

            # prevent log 0 division
            f[np.where(f == 0)] = 1e-17
            if freq == 0:
                freq = 1e-17

            filter[:, :, iOrient] = 2 * np.exp(
                -(1 / 2) * ((np.log2(f) - np.log2(freq)) ** 2 / bw[0] ** 2 + (orient ** 2 / bw[1] ** 2)))

        return filter

    def decomp_Gabor(self, image, degSize=[2, 2], freqRange=[0.5, 20], nFreq=12, nOrient=8, bandwidth=[0.5945, 0.2965]):
        """
        This runs a log-Gabor pyramid decompostion with the given parameters.

        """
        imSize = np.shape(image)
        freq = np.exp(np.linspace(np.log(freqRange[1]), np.log(freqRange[0]), nFreq))
        orient = np.linspace(0, np.pi, (nOrient + 1))
        orient = orient[0:nOrient]

        p = np.exp(np.log(freqRange[1] / freqRange[0]) / nFreq)

        pyr = np.zeros([imSize[0], imSize[1], nOrient, nFreq], dtype='complex128')

        imgFourier = np.fft.fft2(image, axes=(0, 1))
        imgFourier = np.expand_dims(imgFourier, axis=2)

        x = np.zeros((1, imSize[1]))
        y = np.zeros((1, imSize[0]))
        x[0] = np.arange(0, imSize[1])
        y[0] = np.arange(0, imSize[0])
        x = (x - np.ceil(np.mean(x))) / degSize[1]
        y = (y - np.ceil(np.mean(y))) / degSize[0]
        f = np.sqrt((x ** 2) + ((y ** 2).T))
        orient0 = np.arctan2(x, y.T).T

        ## todo: maybe add gradent implementation...
        # pyrGradBW = np.zeros([imSize[0], imSize[1], nOrient, nFreq, 2])

        filters = []
        for iFilter in range(0, nFreq):
            filters.append(np.fft.ifftshift(
                np.fft.ifftshift(self.filter_logGabor(imSize, degSize, freq[iFilter], orient, bandwidth, f, orient0),
                                 0), 1))

            pyr[:, :, :, iFilter] = np.fft.ifft2((filters[iFilter] * imgFourier), axes=(0, 1))

        return pyr, filters, freq

    def V1(self, img, degSize=[2, 2], V1Mode=7, pars=None, Gradidx=None):
        '''
        This runs a log-Gabor pyramid decompostion and computes a normalization of the different channels. Until now,
        only V1Mode 6,7 (local normalization -> only activations at this pixel count) is supported.
        '''

        ## pars
        # current default for V1Mode7 and 1497
        CNaka = 0.  # Naka rushton constant
        ExNaka = 0.  # Naka rushton Exponent below
        ExNakaNorm = 0.  # Naka rushton Exponent below
        bw = [0, 0]
        bw[0] = 0.  # bandwidth in frequency (std of log-Gabor in octaves)
        bw[1] = 0.  # bandwidth in orientation (std log-Gabor in radiants)
        nFreq = 0  # number of frequency bands
        nOrient = 0  # number of orientations
        poolSize = 0
        poolbw = 0  # bandwidth of the normalization pool (Octaves in frequency)
        minF = 0  # lowest frequency band
        maxF = 0  # highest frequency band
        pool0 = 0.

        if pars is not None:
            CNaka = pars[0]
            ExNaka = pars[1] + pars[2]
            ExNakaNorm = pars[1]
            bw = [0, 0]
            bw[0] = pars[3]
            bw[1] = pars[4]
            nFreq = pars[5].astype('int')
            nOrient = pars[6].astype('int')
            poolSize = pars[7].astype('int')
            poolbw = pars[8].astype('int')
            minF = pars[9]
            maxF = pars[10]
            if np.size(pars) >= 12:
                pool0 = pars[11]
        else:
            # Some default parameter for time 1497
            if V1Mode == 6 and pars == None:
                # current default for V1Mode7 and 1497
                CNaka = 0.002689259736807
                ExNaka = 2.169827631936535
                ExNakaNorm = 1.866666687672307
                bw = [0, 0]
                bw[0] = 0.594525260201613
                bw[1] = 0.296469236479833
                nFreq = 12
                nOrient = 8
                poolbw = 1
                minF = 0.5
                maxF = 20
                pool0 = 0.200809948798191
            elif V1Mode == 7 and pars == None:
                # current default for V1Mode7 and 1497
                CNaka = 0.003593813663805
                ExNaka = 4.400000000000000
                ExNakaNorm = 4
                bw = [0, 0]
                bw[0] = 0.594525260201613
                bw[1] = 0.296469236479833
                nFreq = 12
                nOrient = 8
                poolbw = 1
                minF = 0.5
                maxF = 20
                pool0 = 0.111175963679937

        ## Frequency decomposition
        # into log_Gabor frequency and orentation bands
        out, filters, frequencies = decomp_Gabor(img, degSize, [minF, maxF], nFreq, nOrient, bw)
        ao = np.abs(out)

        if V1Mode == 6:
            # HS:
            # spatially pool whole image (Probably only sensible for foveal model)
            # for obvious reasons this ignores the size of the pool in space
            lao = np.log(ao)
            normalizer1 = np.exp(lao * ExNakaNorm)
            normalizer = np.mean(np.mean(normalizer1, 0), 0)  # ToDo: check this line, diff. between mode 6 and 7
            normalizer = np.reshape(normalizer, (1, 1, nOrient, nFreq))

            fdiff = np.linspace(np.log2(minF) - np.log2(maxF), np.log2(maxF) - np.log2(minF),
                                2 * np.size(frequencies) - 1)

            gaussF = np.exp(-fdiff ** 2 / poolbw ** 2 / 2)
            gaussFN = gaussF / np.sum(gaussF)
            gaussFN = np.reshape(gaussFN, [1, 1, 1, np.size(gaussFN)])

            o = np.zeros(nOrient)
            for i in range(0, nOrient):
                o[i] = i - np.floor(nOrient / 2)
            o *= np.pi / nOrient

            # Todo: if (not finite) else if pool0 else:
            gaussO = np.exp(-o ** 2 / pool0 ** 2 / 2)
            gaussON = gaussO / np.sum(gaussO)
            gaussONF = np.fft.fft(np.fft.ifftshift(gaussON))
            gaussONF = np.reshape(gaussONF, (1, 1, nOrient, 1))

            # fourier space filtering to enable "wrap around"
            normalizerF = np.fft.fft(normalizer, axis=2)

            # normalizerF = np.expand_dims(np.expand_dims(normalizerF, 0), 0)
            normalizerF = normalizerF * gaussONF

            normalizer = np.real(np.fft.ifft(normalizerF, axis=2))
            normalizerPad = np.pad(normalizer,
                                   ((0, 0), (0, 0), (0, 0), ((normalizer.shape[3] - 1), (normalizer.shape[3] - 1))),
                                   'constant', constant_values=0)

            normalizer = convolve(normalizerPad, gaussFN, 'valid')
            normalizerFin = (normalizer + CNaka ** ExNakaNorm)
            outNormalized = ((np.exp(lao * ExNaka)) / normalizerFin)

        else:
            # V1Mode7 - for now this is the default.
            # Local normalization -> only activtions at this pixel count
            lao = np.log(ao)
            normalizer = np.exp(lao * ExNakaNorm)

            fdiff = np.linspace(np.log2(minF) - np.log2(maxF), np.log2(maxF) - np.log2(minF),
                                2 * np.size(frequencies) - 1)

            gaussF = np.exp(-fdiff ** 2 / poolbw ** 2 / 2)
            gaussFN = gaussF / np.sum(gaussF)
            gaussFN = np.reshape(gaussFN, [1, 1, 1, np.size(gaussFN)])

            o = np.zeros(nOrient)
            for i in range(0, nOrient):
                o[i] = i - np.floor(nOrient / 2)
            o *= np.pi / nOrient

            # Todo: if (not finite) else if pool0 else:
            gaussO = np.exp(-o ** 2 / pool0 ** 2 / 2)
            gaussON = gaussO / np.sum(gaussO)
            gaussONF = np.fft.fft(np.fft.ifftshift(gaussON))
            gaussONF = np.reshape(gaussONF, (1, 1, np.size(gaussONF)))

            # fourier space filtering to enable "wrap around"
            normalizerF = np.fft.fft(normalizer, axis=2)

            gaussONF = np.expand_dims(gaussONF, 3)
            normalizerF = normalizerF * gaussONF

            normalizer = np.real(np.fft.ifft(normalizerF, axis=2))

            normalizerPad = np.pad(normalizer,
                                   ((0, 0), (0, 0), (0, 0), ((normalizer.shape[3] - 1), (normalizer.shape[3] - 1))),
                                   'constant', constant_values=0)

            normalizer = convolve(normalizerPad, gaussFN, 'valid')
            normalizerFin = (normalizer + CNaka ** ExNakaNorm)
            outNormalized = np.exp(lao * ExNaka) / normalizerFin

            return outNormalized
