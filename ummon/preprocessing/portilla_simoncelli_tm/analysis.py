#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Michael Grunwald at 09.08.2018
"""

from __future__ import division

import numpy as np

from . import filterbank_simoncelli
from .filterbank import Steerable
from .helper import expand, skew2, kurt2, Params


class Analysis:
    """
    Portilla-Simoncelli model's analysis of a texture image by computing different statistics
    that lead to several parameters. Those parameters describe the analyzed texture.
    The parameters can be used later for synthesis.

    Attributes:
        img:            texture image
        nsc:            4 bandpass bands (+hp+lp = 6)
        Na:             spatial neighborhood of autocorrelations is Na x Na coefficients (must be odd)
        Nor:            number of orientations
        pixelStats:     Marginal Statistics of original image
        real:           real part of coefficients
        imag:           imaginary part of coefficients
        magnitude:      magnitude of coefficients
        phase:          phase of coefficients
        im:             image reconstruction from the pyramid beginning with the lowpass residual,
                        then folding up on every scale
        skew0p:         skewness of lp residuals on each level
        kurt0p:         kurtosis of lp residuals on each level
        mag_means:      magnitude coefficient's means per sub-band
        autoCorrReal:   autocorrelation of real coefficients (Na x Na)
        autoCorrMag:    autocorrelation of magnitude coefficients (Na x Na)
        corrMag:        correlation of magnitude sub-bands (between orientations on each scale)
        xCorrMag:       cross-scale correlation of magnitude sub-bands
                        (between orientations of two adjacent scales for each scale)
        corrReal:       correlation of real sub-bands (between orientations on each scale)
        xCorrReal:      cross-scale correlation of real sub-bands
                        (between orientations of two adjacent scales for each scale)
    """

    def __init__(self, img, SP_decomp=True, nsc=4, Na=9, scale_mode=None):
        """
        Initialize all parameters needed to analyze a texture image.

        Args:
            img:            texture image
            sp_decomp:      use Portilla-Simoncelli filter bank
            nsc:            number of scales (bandpasses without high-/low-pass)
            Na:             spatial neighborhood of autocorrelations is Na x Na coefficients (must be odd)
            scale_mode:     default None. 'rescale01', 'norm255'

         Returns:
             Analysis object

        Example:
            >> a = Analysis(im)
            >> stats = a.getFeatures()
        """
        self.scale_mode = scale_mode
        if self.scale_mode == 'norm255':
            self.img / 255
        elif self.scale_mode == 'rescale01':
            self.img = (img - np.min(img)) / (np.max(img) - np.min(img))
        else:
            self.img = img

        self.sp_decomp = SP_decomp
        self.Na = Na
        self.Nsc = nsc
        self.Nor = 4

        self.pixelStats = 0

        B = [len(img) // (2 ** i) for i in range(nsc) if len(img) >= 2 ** i]  # example: [256,128,64]
        # list of bandpasses (with 4 orientations each z,z in dimension B)
        A = [np.array([np.zeros((z, z)).astype(np.complex), np.zeros((z, z)).astype(np.complex),
                       np.zeros((z, z)).astype(np.complex), np.zeros((z, z)).astype(np.complex)]) for z in B]
        hp = [np.array([np.zeros((len(img), len(img)))]).astype(np.complex)]
        lp = [np.array([np.zeros((len(img) // (2 ** 3), len(img) // (2 ** 3)))]).astype(np.complex)]

        # pyramid = hp + A + lp
        self.coeff = hp + A + lp
        self.real = 0
        self.magnitude = 0
        self.LPresidual = 0  # np.zeros((self.real[-1][0].shape))

        self.im = img.copy()  # self.LPresidual.copy()
        self.skew0p = np.zeros((self.Nsc + 1, 1))
        self.kurt0p = np.zeros((self.Nsc + 1, 1))
        nband = sum(len(i) for i in self.coeff)
        self.mag_means = np.zeros((nband))
        self.autoCorrReal = np.zeros((self.Na, self.Na, self.Nsc + 1))
        self.autoCorrMag = np.zeros((self.Na, self.Na, self.Nsc, self.Nor))
        self.corrMag = np.zeros((self.Nor, self.Nor, self.Nsc + 1))  # orientations
        self.xCorrMag = np.zeros((self.Nor, self.Nor, self.Nsc))  # cross-scale
        self.corrReal = np.zeros((2 * self.Nor, 2 * self.Nor, self.Nsc + 1))  # orientations
        self.xCorrReal = np.zeros((2 * self.Nor, 2 * self.Nor, self.Nsc))  # cross-scale

    def setPixelStats(self):
        """
        Compute marginal statistics of the original image.
        """
        mn = self.img.min()
        mx = self.img.max()
        mean = np.mean(self.img)
        var = np.var(self.img)
        skew = skew2(self.img, mean, var)
        kurt = kurt2(self.img, mean, var)
        self.pixelStats = [mean, var, skew, kurt, mn, mx]

    def buildSteerablePyramid(self):
        """
        Build steerable pyramid using one of two pyramid classes.
        """
        if self.sp_decomp:
            coeff_l = filterbank_simoncelli.buildSCFpyr(self.img.copy(), self.Nsc, self.Nor - 1)
        else:
            s = Steerable(self.Nsc + 2)
            coeff_l = s.buildSCFpyr(self.img.copy())

        self.coeff = [np.array(li) for li in coeff_l]

        # mean of residual lp image
        mean_lp = np.mean(self.coeff[-1][0].real)
        # subtract mean of lp residual
        self.coeff[-1][0] = self.coeff[-1][0].real - mean_lp

        # set coefficients
        self.real = [l.real for l in self.coeff.copy()]
        self.magnitude = [np.absolute(l) for l in self.coeff.copy()]

        no = 0
        # subtract mean of magnitude of all sub-bands (including hp and lp residual)
        for lev in range(len(self.magnitude)):
            for ori in range(len(self.magnitude[lev])):
                self.mag_means[no] = np.mean(self.magnitude[lev][ori])
                self.magnitude[lev][ori] = self.magnitude[lev][ori] - self.mag_means[no]
                no = no + 1

    def setAutoCorr_and_PixelStats_of_lpResidual(self):
        """
        Compute autocorrelation and marginal statistics (skewness and kurtosis) of last lp residual.
        """
        self.LPresidual = self.coeff[-1][0].copy()

        if self.sp_decomp:
            coeff2 = filterbank_simoncelli.buildSCFpyr(self.LPresidual.real.copy(), 0, 0)
        else:
            s2 = Steerable(2)  # height including highpass and lowpass
            coeff2 = s2.buildSCFpyr(self.LPresidual.copy())

        # pick lp band (lowpass of lp residual)
        self.im = coeff2[-1][0].real.copy()

        [Nly, Nlx] = self.LPresidual.shape
        Sch = min(Nly, Nlx)
        la = np.floor((self.Na - 1) / 2)
        le = min(Sch / 2 - 1, la)
        cy = Nly / 2 + 1
        cx = Nlx / 2 + 1

        acfft = np.fft.fft2(self.im.copy())
        acabs = np.absolute(acfft) ** 2
        acifft = np.fft.ifft2(acabs).real
        acshift = np.fft.fftshift(acifft)
        ac = np.divide(acshift, self.LPresidual.size)

        # cut off the edges (32x32 --> 9x9)
        ac = ac[int(cy - le - 1):int(cy + le), int(cx - le - 1):int(cx + le)]

        # save in autoCorrReal matrix for the last lp residual image
        self.autoCorrReal[int(la - le):int(la + le + 1), int(la - le):int(la + le + 1), self.Nsc] = ac.copy()

        vari = ac[int(le), int(le)]

        if vari / self.pixelStats[1] > 1e-6:
            self.skew0p[self.Nsc] = np.mean(self.im ** 3) / vari ** 1.5
            self.kurt0p[self.Nsc] = np.mean(self.im ** 4) / vari ** 2
        else:
            self.skew0p[self.Nsc] = 0
            self.kurt0p[self.Nsc] = 3

    def setAutoCorrMag_and_PixelStats_of_bp(self):
        """
        Compute autocorrelation of magnitude coefficients
        and autocorrelation of real parts and marginal statistics (skewness and kurtosis)
        of all lp residual on each level (recursively).
        """

        # coarse-to-fine-loop (bandpass levels without HP/LP/last BP, opposite order!)
        for nsc in range(self.Nsc - 1, -1, -1):
            level = nsc + 1
            for nor in range(len(self.magnitude[level])):
                self.LPresidual = self.magnitude[level][nor].copy()
                [Nly, Nlx] = self.LPresidual.shape
                Sch = min(Nly, Nlx)
                la = np.floor((self.Na - 1) / 2)
                le = min(Sch / 2 - 1, la)
                cy = Nly / 2 + 1
                cx = Nlx / 2 + 1

                acfft = np.fft.fft2(self.LPresidual.copy())
                acabs = np.absolute(acfft) ** 2
                acifft = np.fft.ifft2(acabs).real
                acshift = np.fft.fftshift(acifft)
                ac = np.divide(acshift, self.LPresidual.size)

                # cut off the edges (example: 32x32 --> 9x9)
                ac = ac[int(cy - le - 1):int(cy + le), int(cx - le - 1):int(cx + le)]

                self.autoCorrMag[:, :, nsc, nor] = ac.copy()

            #### Make fake pyramid, containing dummy hp, orientations and dummy lp
            fakePyramid = []
            fakePyramid.append([np.zeros((self.real[level][0].shape))])  # hp
            fakePyramid.append(self.real[level].copy())  # ori
            fakePyramid.append([np.zeros((self.real[level + 1][0].shape))])  # lp

            # reconstruct image of fake coefficient
            if self.sp_decomp:
                fake_image = filterbank_simoncelli.reconSFpyr(fakePyramid.copy(), [1])
            else:
                s4 = Steerable(3)
                fake_image = s4.reconSCFpyr(fakePyramid.copy())

            self.LPresidual = fake_image.copy()

            self.im = np.real(expand(self.im.real.copy(), 2)) / 4
            self.im = self.im + self.LPresidual.copy()

            ac = np.fft.fftshift(np.real(np.fft.ifft2(np.absolute(np.fft.fft2(self.im.copy())) ** 2)))
            ac = ac / (self.LPresidual.size)
            ac = ac[int(cy - le - 1):int(cy + le), int(cx - le - 1):int(cx + le)]

            self.autoCorrReal[int(la - le):int(la + le + 1), int(la - le):int(la + le + 1), nsc] = ac.copy()

            vari = ac[int(le), int(le)]
            if vari / self.pixelStats[1] > 1e-6:
                self.skew0p[nsc] = np.mean(self.im ** 3) / vari ** 1.5
                self.kurt0p[nsc] = np.mean(self.im ** 4) / vari ** 2
            else:
                self.skew0p[nsc] = 0
                self.kurt0p[nsc] = 3

    def setxScalePhaseStats(self):
        """
        Compute cross-scale statistics of real and magnitude coefficients.
        """
        for nsc in range(self.Nsc):
            level = nsc + 1
            bandsize = self.real[level][0].size  # sub-band

            # if not lowpass residual
            if nsc < self.Nsc - 1:
                parents = np.zeros((bandsize, self.Nor))
                rparents = np.zeros((bandsize, self.Nor * 2))
                for nor in range(len(self.real[level])):
                    rtmp = expand(self.coeff[level + 1][nor].real.copy(), 2) / 4
                    itmp = expand(self.coeff[level + 1][nor].imag.copy(), 2) / 4

                    # Double phase:
                    tmp = np.sqrt(rtmp ** 2 + itmp ** 2) * np.exp(2 * np.sqrt(complex(-1)) * np.arctan2(rtmp, itmp))
                    rtmp = tmp.real.copy()
                    itmp = tmp.imag.copy()
                    rparents[:, nor] = rtmp.flatten()
                    rparents[:, self.Nor + nor] = itmp.flatten()
                    tmp = np.absolute(tmp)
                    parents[:, nor] = (tmp - np.mean(tmp)).flatten()
            else:
                tmp = expand(self.real[level + 1][0].copy(), 2).real / 4
                rparents = np.array([tmp.flatten(),
                                     np.roll(tmp, 1, axis=1).flatten(),
                                     np.roll(tmp, -1, axis=1).flatten(),
                                     np.roll(tmp, 1, axis=0).flatten(),
                                     np.roll(tmp, -1, axis=0).flatten()])
                rparents = np.transpose(rparents)
                parents = np.array([])

            cousins = np.reshape(self.magnitude[level].copy(), (self.Nor, bandsize)).T
            nc = cousins.shape[1]
            nparents = parents.shape[1] if parents.ndim > 1 else 0
            self.corrMag[:nc, :nc, nsc] = np.dot(cousins.conj().T, cousins) / bandsize

            if nparents > 0:
                self.xCorrMag[:nc, :nparents, nsc] = np.dot(cousins.conj().T, parents) / bandsize
                if nsc == self.Nsc - 1:  # no possible case ! (because nparents is smaller)
                    self.corrMag[:nparents, :nparents, self.Nsc] = np.dot(parents.conj().T, parents) / (bandsize / 4)

            cousins = np.reshape(self.coeff[level].real.copy(), (self.Nor, bandsize)).T
            nrc = cousins.shape[1]
            nrp = rparents.shape[1]
            self.corrReal[:nrc, :nrc, nsc] = np.dot(cousins.T, cousins) / bandsize

            if nrp > 0:
                self.xCorrReal[:nrc, :nrp, nsc] = np.dot(cousins.T, rparents) / bandsize
                if nsc == self.Nsc - 1:
                    self.corrReal[:nrp, :nrp, self.Nsc] = np.dot(rparents.T, rparents) / (bandsize / 4)

    def computeFeatures(self):
        """
        Compute all statistics.
        """
        self.setPixelStats()
        self.buildSteerablePyramid()

        self.setAutoCorr_and_PixelStats_of_lpResidual()
        self.setAutoCorrMag_and_PixelStats_of_bp()
        self.setxScalePhaseStats()

    def getJointStatisticsFeatures(self, inclPixelStats=True):
        """
        Return computed parameters per statistic of the given texture image.
        """
        channel = self.coeff[0][0]
        varianceHPReal = np.mean(channel ** 2)
        statsLPim = np.asarray([self.skew0p, self.kurt0p])

        if inclPixelStats:
            flattenFeatures = np.concatenate(
                (np.asarray(self.pixelStats).flatten(), statsLPim.flatten(), self.autoCorrReal.flatten(),
                 self.autoCorrMag.flatten(), self.xCorrMag.flatten(), self.corrReal.flatten(),
                 self.xCorrReal.flatten(), varianceHPReal.flatten()))
        else:
            flattenFeatures = np.concatenate(self.autoCorrReal.flatten(),
                                             self.autoCorrMag.flatten(), self.xCorrMag.flatten(),
                                             self.corrReal.flatten(),
                                             self.xCorrReal.flatten(), varianceHPReal.flatten())

        return flattenFeatures

    def getJointStatisticsAsDict(self):
        """
        Return computed parameters per statistic of the given texture image.
        """
        channel = self.coeff[0][0]
        varianceHPReal = np.mean(channel ** 2)
        statsLPim = [self.skew0p, self.kurt0p]
        params_dic = {"pixelStats": self.pixelStats, "pixelLPStats": statsLPim,
                      "autoCorrReal": self.autoCorrReal, "autoCorrMag": self.autoCorrMag,
                      "magMeans": self.mag_means, "cousinMagCorr": self.corrMag,
                      "parentMagCorr": self.xCorrMag, "cousinRealCorr": self.corrReal,
                      "parentRealCorr": self.xCorrReal, "varianceHPR": varianceHPReal}

        params = Params(params_dic)

        return params
