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

    def V1(self, img, degSize=[2, 2], V1Mode=None, pars=None, Gradidx=None):
        '''
        This runs a log-Gabor pyramid decompostion and computes a normalization of the different channels. Until now,
        only V1Mode 7 (local normalization -> only activations at this pixel count) is supported.
        '''

        ## pars
        # Todo: parse pars input.
        CNaka = 0.003593813663805  # Naka rushton constant
        ExNaka = 4.400000000000000  # Naka rushton Exponent below
        ExNakaNorm = 4  # Naka rushton Exponent below
        bw = [0, 0]
        bw[0] = 0.594525260201613  # bandwidth in frequency (std of log-Gabor in octaves)
        bw[1] = 0.296469236479833  # bandwidth in orientation (std log-Gabor in radiants)
        nFreq = 12  # number of frequency bands
        nOrient = 8  # number of orientations
        poolbw = 1  # bandwidth of the normalization pool (Octaves in frequency)
        minF = 0.5  # lowest frequency band
        maxF = 20  # highest frequency band
        pool0 = 0.111175963679937

        ## Frequency decomposition
        # into log_Gabor frequency and orentation bands
        out, filters, frequencies = self.decomp_Gabor(img, degSize, [minF, maxF], nFreq, nOrient, bw)
        ao = np.abs(out)

        ## V1Mode 7 (Ml switch case)
        lao = np.log(ao)
        normalizer = np.exp(lao * ExNakaNorm)

        fdiff = np.linspace(np.log2(minF) - np.log2(maxF), np.log2(maxF) - np.log2(minF), 2 * np.size(frequencies) - 1)

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
