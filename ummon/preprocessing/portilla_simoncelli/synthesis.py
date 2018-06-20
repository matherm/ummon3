from __future__ import division

import impy as ip
import numpy as np

from .config import FILTERPYRAMID, COMPARE_TO_MAT, SEED, MASK, SHAPE, \
        FULLY_COVERED, NITERATION
from . import filterbank_simoncelli
from .auxiliary import forceAutoCorr, adjustCorr1s, \
        adjustCorr2s, signaltonoise, modskew, modkurt, mkAngle, \
        loadFromMat, compare, expand, norm, random_noise
from .filterbank import Steerable


class Synthesis:
    """
    Portilla-Simoncelli model's synthesis synthesizes an arbitrary texture image of any texture
    defined with given parameters.

    """
    def __init__(self, params, niter=NITERATION):
        """

        Args:
            params: parameters that were returned from analysis of any texture
            niter:  number of iterations

        Returns:
            Synthesis object

        """
        params_dic = params.get_dic()
        stats = params_dic["pixelStats"]
        self.mean = stats[0]
        self.var = stats[1]
        self.skew = stats[2]
        self.kurt = stats[3]
        self.mn = stats[4]
        self.mx = stats[5]
        statsLPim = params_dic["pixelLPStats"]
        self.skew0p = statsLPim[0]
        self.kurt0p = statsLPim[1]
        self.varianceHPReal = params_dic["varianceHPR"]
        self.autoCorrReal = params_dic["autoCorrReal"]
        self.autoCorrMag = params_dic["autoCorrMag"]
        self.mag_means = params_dic["magMeans"]
        self.corrMag = params_dic["cousinMagCorr"]
        self.xCorrMag = params_dic["parentMagCorr"]
        #self.corrReal = params_dic["cousinRealCorr"] # not in use
        self.xCorrReal = params_dic["parentRealCorr"]


        if COMPARE_TO_MAT:
            # matlab computed statistics:
            matdict = loadFromMat('params')

            statg0 = np.array(matdict['pixelStats'])
            statsLPim0 = np.array(matdict['pixelLPStats'])
            acr = np.array(matdict['autoCorrReal'])
            ace = np.array(matdict['autoCorrMag'])
            magMeans0 = np.array(matdict['magMeans'])
            C0 = np.array(matdict['cousinMagCorr'])
            Cx0 = np.array(matdict['parentMagCorr'])
            Cr0 = np.array(matdict['cousinRealCorr'])
            Crx0 = np.array(matdict['parentRealCorr'])
            vHPR0 = np.array(matdict['varianceHPR'])

            # compare
            compare(stats, statg0, "Pixel Statistics", rtol=1.e-4)
            compare(statsLPim, statsLPim0, "pixelStats of LP")
            compare(self.autoCorrReal, acr, "autoCorrReal")
            compare(self.autoCorrMag, ace, "autoCorrMag")
            compare(self.mag_means, magMeans0.T, "magMeans")
            compare(self.corrMag, C0, "corrMag")
            compare(self.xCorrMag, Cx0, "CrossCorrMag")
            compare(self.corrReal, Cr0, "CorrReal")
            compare(self.xCorrReal, Crx0, "CrossCorrReal")
            compare(self.varianceHPReal, vHPR0, "varianceHPReal")

        # extract configuration sizes of the underlying analysis from the parameters
        tmp = self.autoCorrMag.shape
        self.Na = tmp[0] # Spatial neighborhood is Na x Na coefficients (odd number)
        self.Nsc = tmp[2]
        self.Nor = tmp[len(tmp)-1]*(len(tmp)==4) + (len(tmp)<4)
        self.im = np.array([])
        self.Niter = niter #12 # Number of iterationen
        self.p = 1

        #initialize
        self.prev_im = np.array([])
        self.nq = 0
        self.Nq = np.floor(np.log2(self.Niter))
        self.imS = np.array([])
        Nband = self.Nsc*self.Nor+2
        self.snr1 = np.zeros((self.Niter,Nband))
        self.snr2 = np.zeros((self.Niter,self.Nsc+1))
        self.snr3 = np.zeros((self.Niter,self.Nsc))
        self.snr4 = np.zeros((self.Niter,self.Nsc))
        self.snr6 = np.zeros((self.Niter))
        self.snr7 = np.zeros((self.Niter, 2*(self.Nsc+1)+5)) #  2*(Nsc+1)+4 out of boundary ?!
        self.snr4r = np.zeros((self.Niter,self.Nsc))

        #B = [len(self.img)//(2**i) for i in range(self.Nsc) if len(self.img) >= 2**i ] # example: [256,128,64]
        #A = [np.array([np.zeros((z, z)).astype(np.complex), np.zeros((z, z)).astype(np.complex),
        #    np.zeros((z, z)).astype(np.complex), np.zeros((z, z)).astype(np.complex)]) for z in B]
        #  list of bandpasses (with 4 orientations each z,z in dimension B)
        #hp = [np.array([np.zeros((len(img), len(img)))]).astype(np.complex)]
        #lp = [np.array([np.zeros((len(img)//(2**3), len(img)//(2**3)))]).astype(np.complex)]
        #pyramid = hp + A + lp

        self.coeff = np.array([])
        self.real = 0
        self.imag = 0
        self.magnitude = 0
        self.phase = 0


    def initialWhiteNoise(self, nsize, random):
        """
        Generate a random noise image as initial image for synthesis process.
        """

        # to use an initial random image to initial defects
        if type(random) == np.ndarray and random.ndim > 1:
            if FULLY_COVERED:
                self.im = norm(random)
            else:
                self.im = ip.scale_vals(random, 0, 1)

        elif SEED:
            # matlab seed to compare with matlab
            self.im = ip.read_img("../matlab/textureSynth/random.png")
            print('use seed as initial random noise.')
            if FULLY_COVERED:
                self.im = norm(self.im)
            else:
                self.im = ip.scale_vals(self.im, 0, 1)


        else:
            self.im = random_noise(self.mean, self.var, nsize)


        [Ny, Nx] = self.im.shape
        #self.Nq = np.floor(np.log2(self.Niter))

        nth = np.log2(min(Ny,Nx)/self.Na)
        if nth < self.Nsc+1:
            print("Warning: Na will be cut off for levels above ", np.floor(nth))

        self.im = (self.im).astype(np.float64)
        self.prev_im = self.im.copy()

        self.imS = np.zeros((self.im.shape[0],self.im.shape[1],int(self.Nq)+1))


    def buildSteerablePyramid(self):
        """
        Compute the steerable pyramid coefficients of the present image.
        """
        if FILTERPYRAMID:
            coeff_list = filterbank_simoncelli.buildSCFpyr(self.im.copy(),self.Nsc,self.Nor-1)
        else:
            s = Steerable(self.Nsc+2) # Nsc(bandpass) + hp + lp
            coeff_list = s.buildSCFpyr(self.im.copy())

        # convert list of list of numpy array to list of numpy array of numpy array
        self.coeff = [np.array(li) for li in coeff_list]


    def forceStatsLP(self,niter):
        """
        Force the autocorrelation real, skewness and kurtosis to the lowpass residual.
        """
        # (2) Subtract mean of lowBand
        mean_lp = np.mean(self.coeff[-1][0]) #!!!! coeff LP is real in matlab .. here it is complex (also different values in coeff) /!\
        # subtract mean
        self.coeff[-1][0] = self.coeff[-1][0] - mean_lp

        #self.real = [l.real for l in self.coeff.copy()]
        self.magnitude = [np.absolute(l) for l in self.coeff.copy()]

        # (3) Adjust autoCorr of lowBand
        lpResidual = self.coeff[-1][0].copy()

        Sch = min([lpResidual.shape[0], lpResidual.shape[1]])/2
        nz = np.sum(np.logical_not(np.isnan(self.autoCorrReal[:,:,self.Nsc])).astype(np.int))
        la = (self.Na-1)/2
        lz = ((np.sqrt(nz)-1)/2).astype(np.int)
        le = int(min([Sch/2-1,lz]))

        # reconstructed image: initialize to lowBand
        self.im = lpResidual.real.copy()

        if FILTERPYRAMID:
            coeff2 = filterbank_simoncelli.buildSCFpyr(self.im.copy(),0,0)
        else:
            s2 = Steerable(2) #height including highpass and lowpass
            coeff2 = s2.buildSCFpyr(self.im.copy())

        self.im = coeff2[-1][0].copy() # only lp of lp residuals

        vari = self.autoCorrReal[int(la),int(la),int(self.Nsc)].copy() # autoCorrReal[la:la+1,la:la+1,Nsc]

        if vari/self.var > 1e-4:
            # 2 statistics - correlation of subbands
            [self.im,self.snr2[niter,self.Nsc],Chf] = forceAutoCorr(self.im.copy(), self.autoCorrReal[:,:,self.Nsc].copy(), self.p)
            self.im = self.im.real.copy()

            # 1 statistics - marginal statistics
            [self.im, self.snr7[niter, 2*(self.Nsc+1)-2]] = modskew(self.im.copy(), self.skew0p[self.Nsc].copy(), self.p) #adjusts skewness
            [self.im, self.snr7[niter, 2*(self.Nsc+1)-1]] = modkurt(self.im.copy(), self.kurt0p[self.Nsc].copy(), self.p) #adjusts kurtosis

        else:
            self.im = self.im * np.sqrt(np.complex(vari/np.var(self.im)))
            self.im = self.im.real.copy()

        if np.var(lpResidual.imag) / np.var(lpResidual.real) > 1e-6:
            print("Discarding non-trivial imaginary part, lowPass autoCorr!")


    def subtractMagMean(self):
        """
        Subtract the means of each magnitude sub-band.
        """
        # (4) Subtract mean of magnitude
        # subtract mean of magnitude of all bands (including hp and lp residual)
        for lev in range(len(self.magnitude)):
            for ori in range(len(self.magnitude[lev])):
                current_mag_mean = np.mean(self.magnitude[lev][ori])
                self.magnitude[lev][ori] = self.magnitude[lev][ori] - current_mag_mean


    def forceCoarseToFine(self,niter):
        """
        Force statistics in coarse-to-fine loop.
        """
        # (5) Coarse-to-fine loop
        for nsc in range(self.Nsc-1,-1,-1):
            print("niter: ", niter, "nsc: ", nsc)
            level = nsc+1
            bandlen = len(self.coeff[level][0])
            bandsize = self.coeff[level][0].size

            # (a) interpolate parents
            if nsc < self.Nsc-1:
                parents = np.zeros((bandsize, self.Nor))
                rparents = np.zeros((bandsize, self.Nor*2))
                for nor in range(self.Nor):
                    # upscale + one orientation is 1/4 part
                    # real + imag
                    rtmp = expand(self.coeff[level+1][nor].real,2)/4
                    itmp = expand(self.coeff[level+1][nor].imag,2)/4
                    tmp = np.sqrt(rtmp**2 + itmp**2) * np.exp(2*np.sqrt(np.complex(-1))*np.arctan2(rtmp,itmp))
                    rtmp = np.copy(tmp.real)
                    itmp = np.copy(tmp.imag)
                    rparents[:,nor] = rtmp.flatten()
                    rparents[:,self.Nor+nor] = itmp.flatten()
                    # magnitude
                    tmp = np.absolute(tmp)
                    tmp = tmp - np.mean(tmp)
                    parents[:,nor] = tmp.flatten()
            else: # last lp residual --> no need to interpolate on this level (lp residual is not split into 4 orientations)
                rparents = np.array([])
                parents = np.array([])

            # (b) Adjust cross-correlation with MAGNITUDES at other orientations/scales
            cousins = np.reshape(self.magnitude[level].copy(), (self.Nor,bandsize)).T
            nc = cousins.shape[1]
            nparents = parents.shape[1] if parents.ndim > 1 else 0

            if nparents == 0:
                [cousins, self.snr3[niter,nsc], M] = adjustCorr1s(cousins.copy(), self.corrMag[:nc,:nc,nsc].copy(), 2, self.p)
            else:
                [cousins, self.snr3[niter,nsc], self.snr4[niter,nsc], Mx, My] = adjustCorr2s(cousins.copy(), self.corrMag[:nc,:nc,nsc].copy(), parents.copy(), self.xCorrMag[:nc,:nparents,nsc].copy(), 3, self.p)

            if np.var(cousins.imag)/np.var(cousins.real) > 1e-6:
                print('Non-trivial imaginary part, xCorrMag, lev=', nsc, '!\n')
            else:
                cousins = cousins.real
                self.magnitude[level] = np.reshape(cousins.copy().T,(self.Nor,bandlen,bandlen))

            # (c) Adjust autoCorr of mag responses
            Sch = np.minimum(self.magnitude[level][0].shape[0]/2, self.magnitude[level][0].shape[1]/2)
            nz = np.sum(np.logical_not(np.isnan(self.autoCorrMag[:,:,nsc,0])).astype(int))
            la = (self.Na-1)/2
            lz = (np.sqrt(np.complex(nz))-1)/2
            le = np.minimum(Sch/2-1,lz)
            for nor in range(len(self.magnitude[level])):
                # avoid nband by changing indicies from mag_means into 2 dims
                nband = 1 + nsc*self.Nor + nor
                ch = self.magnitude[level][nor].copy()
                [ch, self.snr1[niter, nband], Chf] = forceAutoCorr(ch.copy(), self.autoCorrMag[int(la-le):int(la+le+1),int(la-le):int(la+le+1),int(nsc),int(nor)].copy(), self.p)
                ch = ch.real
                self.magnitude[level][nor] = ch.copy()

                # Impose magnitude
                mag = self.magnitude[level][nor].copy() + self.mag_means[nband]
                mag[mag<0] = 0
                absolute = np.absolute(self.coeff[level][nor].copy())
                absolute[absolute<np.spacing(1)] += 1
                self.coeff[level][nor] = np.multiply(self.coeff[level][nor], np.divide(mag, absolute))

            # (d) Adjust cross-correlation of REAL PARTS at other orientations/scales
            cousins = np.reshape(self.coeff[level].real.copy(), (self.Nor,bandsize)).T
            Nrc = cousins.shape[1]
            Nrp = rparents.shape[1] if rparents.ndim > 1 else 0

            if Nrp != 0:
                a3 = 0
                a4 = 0
                for nrc in range(Nrc):
                    cou = cousins[:,nrc].copy()
                    [cou, s3, s4, Mx, My] = adjustCorr2s(np.transpose(np.atleast_2d(cou.copy())), np.mean(cou.copy()**2), rparents.copy(), self.xCorrReal[nrc,:Nrp,nsc].copy(), 3, self.p) #p # atleast_2d to get 2 dimensional vector
                    a3 = s3 + a3
                    a4 = s4 + a4
                    cousins[:,nrc] = cou[:,0].copy()

                self.snr4r[niter,nsc] = a4/Nrc

            if (np.var(cousins.imag) / np.var(cousins.real)) > 1e-6:
                print('Non-trivial imaginary part, real crossCorr, lev=', nsc, '!\n')
            else:
                # NOTE: THIS SETS REAL PART ONLY - signal is now NONANALYTIC!
                self.coeff[level] = np.reshape(cousins.copy().T,(self.Nor,bandlen,bandlen)).astype(np.complex_)

            # (e) Re-create analytic subbands
            dims = self.coeff[level][0].shape
            ctr = np.ceil(((dims[0]+0.5)/2,(dims[1]+0.5)/2))
            ang = mkAngle(dims, 0, ctr)
            ang[int(ctr[0]-1),int(ctr[1]-1)] = -np.pi/2
            for nor in range(self.Nor):
                ch = self.coeff[level][nor].copy()
                ang0 = np.pi*(nor)/self.Nor
                xang = np.mod((ang - ang0 + np.pi), 2*np.pi) - np.pi
                amask = 2*(np.absolute(xang) < np.pi/2) + (np.absolute(xang) == np.pi/2)
                amask[int(ctr[0]-1),int(ctr[1]-1)] = 1
                amask[:,0] = 1
                amask[0,:] = 1
                amask = np.fft.fftshift(amask.copy())
                ch = np.fft.ifft2(np.multiply(amask, np.fft.fft2(ch.copy())))
                self.coeff[level][nor] = ch.copy()

            # (f) Combine ori bands

            # (g) Make fake pyramid, containing dummy hi, ori, lo # ?!?! fake pyramid with orientations etc. in right scale?
            fakePyramid = []
            fakePyramid.append([np.zeros(self.coeff[level][0].shape)]) #hp
            fakePyramid.append(self.coeff[level].real.copy()) #ori
            fakePyramid.append([np.zeros(self.coeff[level+1][0].shape)]) #lp

            # (h) reconstruct image of fake coefficient
            if FILTERPYRAMID:
                fake_image = filterbank_simoncelli.reconSFpyr(fakePyramid.copy(), [1]) #!!! height=1
            else:
                s4 = Steerable(3)
                fake_image = s4.reconSCFpyr(fakePyramid.copy())

            ch = fake_image.copy()
            self.im = np.real(expand(self.im,2))/4
            self.im = self.im + ch

            vari = self.autoCorrReal[int(la),int(la),int(nsc)].copy()
            if vari/self.var > 1e-4:
                [self.im, self.snr2[niter, nsc], Chf] = forceAutoCorr(self.im.copy(), self.autoCorrReal[int(la-le):int(la+le+1),int(la-le):int(la+le+1),int(nsc)].copy(), self.p)
            elif vari/self.var == 0.:
                print('Variance difference is too large.')
            else:
                self.im = self.im * np.sqrt(np.complex(np.divide(vari,np.var(self.im))))

            self.im = self.im.real
            # (i) Fix marginal stats
            if vari/self.var > 1e-4:
                [self.im, self.snr7[niter, 2*nsc-1]] = modskew(self.im.copy(), self.skew0p[nsc].copy(), self.p) #adjusts skewness
                [self.im, self.snr7[niter, 2*nsc]] = modkurt(self.im.copy(), self.kurt0p[nsc].copy(), self.p) #adjusts kurtosis

    def forceStatsHP(self):
        """
        Force the statistics on the highpass response.
        """
        # (6) Adjust variance in HP, if higher than desired
        ch = self.coeff[0][0].copy()
        vHPR = np.mean(ch**2)
        if vHPR > self.varianceHPReal:
            ch = np.multiply(ch.copy(), np.sqrt(np.complex(np.divide(self.varianceHPReal,vHPR))))
            self.coeff[0][0] = ch.copy()

        # real parts of pyramid
        real_coeff = [l.real for l in self.coeff.copy()]

        if FILTERPYRAMID:
            hp = filterbank_simoncelli.reconSFpyr(real_coeff.copy(), [0]) #!!! height=0
        else:
            s5 = Steerable(2)
            hp = s5.reconSCFpyr(real_coeff.copy()) # reconstruct hp only
        self.im = self.im + hp.copy()


    def forcePixelStats(self,niter, imask):
        """
        Force the pixel statistics on the reconstructed image.
        """
        # (7) Pixel statistics
        means = np.mean(self.im) # mean difference python: -1.9e-13 to matlab: -3.47e-16
        variance = np.var(self.im, dtype=np.float64)
        self.snr7[niter, 2*(self.Nsc+1)+1] = signaltonoise(self.var, self.var-variance)
        self.im = self.im-means
        mns = np.min(self.im+self.mean)
        mxs = np.max(self.im+self.mean)
        self.snr7[niter, 2*(self.Nsc+1)+2] = signaltonoise(self.mx-self.mn, np.sqrt((self.mx-mxs)**2+(self.mn-mns)**2))
        self.im = self.im * np.sqrt(np.complex(np.divide(((1-self.p)*variance + self.p*self.var),variance)))
        self.im = self.im + self.mean

        # Adjusts skewness (keep mean and variance)
        [self.im, self.snr7[niter, 2*(self.Nsc+1)+2]] = modskew(self.im.copy(), self.skew, self.p)
        # Adjusts kurtosis (keep mean and variance, but not skewness)
        [self.im, self.snr7[niter, 2*(self.Nsc+1)+3]] = modkurt(self.im.copy(), self.kurt, self.p)
        # adjusts range (affects everything)
        self.im = np.maximum(np.minimum(self.im,(1-self.p)*np.max(self.im)+self.p*self.mx), \
                             (1-self.p)*np.min(self.im)+self.p*self.mn)
        ##else (without marginal statistics ?  - left out)
        #snr7[niter,2*(Nsc+1)+3] = signaltonoise(skew,skew-skew2(im, np.mean(im), np.var(im)))
        #snr7[niter,2*(Nsc+1)+4] = signaltonoise(kurt,kurt-kurt2(im, np.mean(im), np.var(im)))

        # (8) Force pixels specified by image mask
        if MASK:
            self.im = np.multiply(imask[:,:,0], imask[:,:,1]) + np.multiply(1-imask[:,:,0],self.im)

        self.snr6[niter] = signaltonoise(self.im.copy()-self.mean, self.im.copy()-self.prev_im.copy())

        if np.floor(np.log2(niter+1)) == np.log2(niter+1):
            self.imS[:,:,self.nq] = self.im.copy()
            self.nq = self.nq + 1

        tmp = self.prev_im.copy()
        self.prev_im = self.im.copy()

        #accelerator
        alpha = 0.8
        self.im = self.im + alpha*(self.im - tmp)


    def forceStatistics(self, imask, width):
        """
        Compute all statistics.
        """
        for niter in range(self.Niter):
            print("NITER: ", niter)
            self.buildSteerablePyramid()
            self.forceStatsLP(niter)
            self.subtractMagMean()
            self.forceCoarseToFine(niter)
            self.forceStatsHP()
            self.forcePixelStats(niter, imask)


    def synthesizeImage(self, nsize=SHAPE, random=0, imask=np.array([0]), width=False):
        """
        Synthesize an image with a random white noise image.
        """
        # ToDo: move iteration number to this function
        [Ny, Nx] = nsize

        if  not (imask==0).all():
            if imask[:,:,0].size != Ny*Nx:

                print('imask size ',imask[:,:,0].shape ,'does not match image dimensions ', nsize)

            if imask.shape[2] == 1:
                a = (np.cos*np.array([i for i in range(-np.pi/2,np.pi*(1-2/Ny)/2,2*np.pi/Ny)])).T
                b = np.cos(np.array([i for i in range(-np.pi/2,np.pi*(1-2/Nx)/2,2*np.pi/Nx)]))
                mask = a * b
                mask = mask**2
                aux = np.zeros(nsize)
                aux[Ny/4,Ny/4+Ny/2,Nx/4:Nx/4+Nx/2] = mask
                mask = aux
            else:
                mask = imask[:,:,0]

        self.initialWhiteNoise(nsize, random)
        self.forceStatistics(imask, width)

    def getImage(self):
        """
        Get synthesized image of the texture.
        """
        if FULLY_COVERED:
            im = norm(self.im.real)
        else:
            im = ip.scale_vals(self.im.real, 0, 1)

        return im

