from __future__ import division
import numpy as np
from scipy.special import factorial


def buildSCFpyr(im, ht=-1, order=3, twidth=1):

    # default
    max_ht = np.floor(np.log2(np.min(im.shape))+2)

    if ht == -1:
        ht = max_ht
        print("ht: ", ht)
    else:
        if ht > max_ht:
            print('Cannot build pyramid higher than ',max_ht,' levels.')

    nbands = order + 1

    # Steering stuff:
    if np.mod(nbands,2) == 0:
        harmonics = np.array([i for i in range(0,np.int(nbands/2))]).T * 2 + 1
    else:
        harmonics = np.array([i for i in range(0,np.int((nbands-1)/2+1))]).T * 2

    steermtx = steer2HarmMtx(harmonics, np.pi*np.array([i for i in range(0,nbands)])/nbands, 'even') #-1, 'even'

    #----------------------------------------------------------------

    dims = im.shape
    ctr = np.ceil(np.array([dims[0]+0.5, dims[1]+0.5])/2)

    m = np.divide(np.array([i for i in range(1,dims[1]+1)])-ctr[1],dims[1]/2)
    n = np.divide(np.array([i for i in range(1,dims[0]+1)])-ctr[0],dims[0]/2)
    [xramp,yramp] = np.meshgrid(m,n)                    
    angle = np.arctan2(yramp,xramp)
    log_rad = np.sqrt(xramp**2 + yramp**2)
    log_rad[int(ctr[0]-1),int(ctr[1]-1)] =  log_rad[int(ctr[0]-1),int(ctr[1]-2)]
    log_rad  = np.log2(log_rad)

    # Radial transition function (a raised cosine in log-frequency):
    [Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),np.array([0, 1]))
    Yrcos = np.sqrt(Yrcos)

    YIrcos = np.sqrt(1.0 - Yrcos**2)
    lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

    ## to save single filter masks
    #path_name = "result/filter_masks/lo0mask_" + str(ht)+"-"+ str(order) + ".png"
    #ip.save_img(lo0mask, path_name)

    imdft = np.fft.fftshift(np.fft.fft2(im))
    lo0dft =  np.multiply(imdft,lo0mask)

    pyr = buildSCFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands)

    hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0])
    ## to save single filter masks
    #path_name = "result/filter_masks/hi0mask_"+ str(ht)+"-"+str(order)+ ".png"
    #ip.save_img(hi0mask, path_name)

    hi0dft =  np.multiply(imdft, hi0mask)
    hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

    ret_pyr = []
    ret_pyr.append([hi0.real])

    for b in pyr:
        ret_pyr.append(b)

    return ret_pyr #, steermtx, harmonics


def buildSFpyr(im, ht=-1, order=3, twidth=1):
    """
    [pyr,pind,steermtx,harmonics] = buildSFpyr(im, ht, order, twidth)
    """
    # DEFAULTS:

    max_ht = np.floor(np.log2(np.min(im.shape))) - 2 #SCFpyr: log2(min(size(im)))+2)

    if ht == -1:
        ht = max_ht
    else:
        if ht > max_ht:
            print('Cannot build pyramid higher than ',max_ht,' levels.')


    if (order > 15) or (order < 0):
      print('Warning: ORDER must be an integer in the range [0,15]. Truncating.')
      order = min(max(order,0),15) # ??
    else:
      order = np.round(order)

    nbands = order+1

    if twidth <= 0:
      print('Warning: TWIDTH must be positive.  Setting to 1.')
      twidth = 1

    # Steering stuff:
    if np.mod((nbands),2) == 0:
        harmonics = np.array([i for i in range(0,np.int(nbands/2))]).T * 2 + 1
    else:
        harmonics = np.array([i for i in range(0,np.int((nbands-1)/2+1))]).T * 2

    steermtx = steer2HarmMtx(harmonics, np.pi*np.array([i for i in range(0,nbands)])/nbands, 'even')

    #-----------------------------------------------------------------

    dims = im.shape
    ctr = np.ceil(np.array([dims[0]+0.5, dims[1]+0.5])/2)
    m = np.divide(np.array([i for i in range(0,dims[1])])-ctr[1],dims[1]/2)
    n = np.divide(np.array([i for i in range(0,dims[0])])-ctr[0],dims[0]/2)
    [xramp,yramp] = np.meshgrid(m,n)
    angle = np.arctan2(yramp,xramp)
    log_rad = np.sqrt(xramp**2 + yramp**2)
    log_rad[ctr[0]-1,ctr[1]-1] =  log_rad[ctr[0]-1,ctr[1]-2]
    log_rad  = np.log2(log_rad)

    # Radial transition function (a raised cosine in log-frequency):
    [Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),np.array([0, 1]))
    Yrcos = np.sqrt(Yrcos)

    YIrcos = np.sqrt(1.0 - Yrcos**2)
    lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

    ## to save single filter masks
    #path_name = "result/filter_masks/lo0mask_" + str(ht)+"-"+ str(order) + ".png"
    #ip.save_img(lo0mask, path_name)

    imdft = np.fft.fftshift(np.fft.fft2(im))
    lo0dft =  np.multiply(imdft, lo0mask)

    pyr = buildSFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands)

    hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0])
    ## to save single filter masks
    #path_name = "result/filter_masks/hi0mask_"+ str(ht)+"-"+str(order)+ ".png"
    #ip.save_img(hi0mask, path_name)
    hi0dft =  np.multiply(imdft, hi0mask)
    hi0 = np.fft.ifft2(np.fft.ifftshift(hi0dft))

    ret_pyr = []
    ret_pyr.append([hi0.real])

    for b in pyr:
        ret_pyr.append(b)

    return ret_pyr #, steermtx, harmonics]



def buildSCFpyrLevs(lodft,log_rad,Xrcos,Yrcos,angle,ht,nbands):
    """
    Returns:

    function [pyr,pind] = buildSCFpyrLevs(lodft,log_rad,Xrcos,Yrcos,angle,ht,nbands);
    """

    if ht <= 0:
        lo0 = np.fft.ifft2(np.fft.ifftshift(lodft))
        pyr = [[lo0.real]]
    else:
        orients = []

        log_rad = log_rad + 1

        lutsize = 1024
        Xcosn = np.pi*np.array([i for i in range(-(2*lutsize+1),(lutsize+1)+1)])/lutsize  # [-2*pi:pi]
        order = nbands-1
        # divide by sqrt(sum_(n=0)^(N-1)  cos(pi*n/N)^(2(N-1)) )
        const =  np.divide(np.multiply(2**(2*order),factorial(order)**2),nbands*factorial(2*order))

        #  Ycosn = sqrt(const) * (cos(Xcosn)).^order;
        # analityc version: only take one lobe
        alfa = np.mod(np.pi+Xcosn,2*np.pi)-np.pi
        Ycosn = 2*np.sqrt(const) * np.multiply((np.cos(Xcosn)**order), (np.absolute(alfa)<np.pi/2))

        himask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

        #ori = 0
        for b in range(0,nbands):
            anglemask = pointOp(angle, Ycosn, Xcosn[0]+np.pi*(b)/nbands, Xcosn[1]-Xcosn[0])
            banddft = np.multiply(np.multiply(np.multiply(((-1j)**(nbands-1)), lodft), anglemask), himask)
            band = np.fft.ifft2(np.fft.ifftshift(banddft))

            #band_mask =  anglemask * himask
            #path_name = "result/filter_masks/band_mask"+ str(ht)+"-"+ str(ori) + ".png"
            #ip.save_img(band_mask, path_name)
            #ori = ori +1

            #  bands(:,b) = real(band(:));
            # analytic version: full complex value

            #bands[:,:,b] = band
            orients.append(band)
            #bind[b,:]  = size(band);

        dims = lodft.shape
        ctr = np.ceil(np.array([dims[0]+0.5, dims[1]+0.5])/2)
        #ctr = np.ceil((dims+0.5)/2)
        lodims = np.ceil(np.array([dims[0]-0.5, dims[1]-0.5])/2)
        #lodims = np.ceil((dims-0.5)/2)
        loctr = np.ceil((lodims+0.5)/2)
        lostart = ctr-loctr
        loend = lostart+lodims

        log_rad = log_rad[int(lostart[0]):int(loend[0]),int(lostart[1]):int(loend[1])]
        angle = angle[int(lostart[0]):int(loend[0]),int(lostart[1]):int(loend[1])]
        lodft = lodft[int(lostart[0]):int(loend[0]),int(lostart[1]):int(loend[1])]
        YIrcos = np.absolute(np.sqrt(1.0 - Yrcos**2))
        lomask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

        lodft = np.multiply(lomask,lodft)

        npyr = buildSCFpyrLevs(lodft, log_rad, Xrcos, Yrcos, angle, ht-1, nbands)

        pyr = []
        pyr.append(orients)
        for b in npyr:
            pyr.append(b)

    return pyr


def reconSFpyr(pyr, levs):
    """
    Returns:
        res
    """

    nbands = len(pyr[1]) # number orientations

    dims = pyr[0][0].shape

    ctr = np.ceil(np.array([dims[0]+0.5, dims[1]+0.5])/2)

    m = np.divide(np.array([i for i in range(1,dims[1]+1)])-ctr[1],dims[1]/2)
    n = np.divide(np.array([i for i in range(1,dims[0]+1)])-ctr[0],dims[0]/2)

    [xramp,yramp] = np.meshgrid(m,n)
    angle = np.arctan2(yramp,xramp)
    log_rad = np.sqrt(xramp**2 + yramp**2)
    log_rad[int(ctr[0]-1),int(ctr[1]-1)] = log_rad[int(ctr[0]-1),int(ctr[1]-2)]
    log_rad  = np.log2(log_rad)

    # Radial transition function (a raised cosine in log-frequency):
    [Xrcos,Yrcos] = rcosFn(1,(-1/2),np.array([0, 1]))
    Yrcos = np.sqrt(Yrcos)
    YIrcos = np.sqrt(np.absolute(1.0 - Yrcos**2))

    z = 0
    for subband in pyr:
        for band in subband:
            z += 1

    if (z == 2):
        if ((levs==1).any()):
            resdft = np.fft.fftshift(np.fft.fft2(pyr[1][0]))
        else:
            resdft = np.zeros((pyr[1][0].shape))
    else:
        resdft = reconSFpyrLevs(pyr[1:].copy(), log_rad, Xrcos, Yrcos, angle, levs, nbands) # levs / [1]


    lo0mask = pointOp(log_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

    #save lo0mask in fourier and image domain
    #path_name = "result/filter_masks/lo0mask.png"
    #ip.save_img(lo0mask, path_name)
    #path_name = "result/filter_masks/lo0mask_wavelet.png"
    #ip.save_img(np.fft.ifft2(lo0mask), path_name)

    resdft = np.multiply(resdft, lo0mask)

    # residual highpass subband
    if (np.array(levs) == 0).any():
        hi0mask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0])
        hidft = np.fft.fftshift(np.fft.fft2(pyr[0][0].copy()))

        ## save hi0mask in fourier and image domain
        #path_name = "result/filter_masks/hi0mask.png"
        #ip.save_img(hi0mask, path_name)

        resdft = resdft + np.multiply(hidft, hi0mask)

    res = np.real(np.fft.ifft2(np.fft.ifftshift(resdft)))

    return res



def reconSFpyrLevs(pyr,log_rad,Xrcos,Yrcos,angle,levs,nbands):
    """
    Returns:
        resdft
    """

    lo_ind = nbands + 1
    dims = pyr[0][0].shape
    ctr = np.ceil(np.array([dims[0]+0.5,dims[1]+0.5])/2)

    # log_rad = log_rad + 1;
    Xrcos = Xrcos - np.log2(2) # shift origin of lut by 1 octave.

    if (np.array(levs) > 1).any():
        lodims = np.ceil(np.array([dims[0]-0.5,dims[1]-0.5]) / 2)
        loctr = np.ceil((lodims + 0.5) / 2)
        lostart = ctr - loctr + 1
        loend = lostart + lodims - 1
        nlog_rad = log_rad[lostart[0]-1:loend[0],lostart[1]-1:loend[1]]
        nangle = angle[lostart[0]-1:loend[0],lostart[1]-1:loend[1]]

        z = 0
        for band in pyr:
            for subband in band:
                z += 1

        if z>lo_ind:
            nresdft = reconSFpyrLevs( pyr[1:].copy(), nlog_rad, Xrcos, Yrcos, nangle,np.array(levs)-1, nbands)
        else:
            nresdft = np.fft.fftshift(np.fft.fft2(pyr[1][0].copy()))

        YIrcos = np.sqrt(np.absolute(1.0 - Yrcos**2))
        lomask = pointOp(nlog_rad, YIrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

        resdft = np.zeros((dims))
        resdft[lostart[0]-1:loend[0],lostart[1]-1:loend[1]] = np.multiply(nresdft, lomask) #?? complex to real cast?
    else:
        resdft = np.zeros((dims))

    if (np.array(levs)==1).any():
        lutsize = 1024
        Xcosn = np.pi * np.array([i for i in range(-(2*lutsize+1),(lutsize+2))])/lutsize  # [-2*pi:pi]
        order = nbands - 1
        #%% divide by sqrt(sum_(n=0)^(N-1)  cos(pi*n/N)^(2(N-1)) )
        const = np.multiply((2**(2*order)), np.divide((factorial(order)**2),(nbands*factorial(2*order))))
        Ycosn = np.sqrt(const) * (np.cos(Xcosn))**order
        himask = pointOp(log_rad, Yrcos, Xrcos[0], Xrcos[1]-Xrcos[0])

        for b in range(0,nbands):
            #if (bands==b).any:
            anglemask = pointOp(angle,Ycosn,Xcosn[0]+np.pi*(b)/nbands,Xcosn[1]-Xcosn[0])
            banddft = np.fft.fftshift(np.fft.fft2(pyr[0][b].copy()))
            resdft = resdft + (np.sqrt(complex(-1)))**(nbands-1) * np.multiply(np.multiply(banddft,anglemask),himask)
            #end

    return resdft


def steer2HarmMtx(harmonics, angles=-1, evenorodd='even'):
    """
    mtx = steer2HarmMtx(harmonics, angles, evenorodd)
    """
    # Make HARMONICS a row vector
    #print("harmonics.shape: ", harmonics.shape)
    #harmonics = harmonics.T

    numh = 2 * np.atleast_2d(harmonics).shape[1] - (harmonics == 0).any()

    #if angles == -1:
    #    angles = np.pi * np.array([i for i in range(0,numh)]).T/numh


    #=================================================================

    if evenorodd == 'even':
        evenorodd = 0
    elif evenorodd == 'odd':
        evenorodd = 1
    else:
        print('EVEN_OR_ODD should be the string  EVEN or ODD')


    # Compute inverse matrix, which maps Fourier components onto
    # steerable basis.
    imtx = np.zeros((angles.shape[0],numh))
    col = 0
    for h in harmonics:
        args = h * angles
        if h == 0:
            imtx[:,col] = np.ones((angles.shape))
            col = col+1
        elif evenorodd:
            imtx[:,col] = np.sin(args)
            imtx[:,col+1] = -np.cos(args)
            col = col+2
        else:
            imtx[:,col] = np.cos(args)
            imtx[:,col+1] = np.sin(args)
            col = col + 2


    r = rank(imtx)
    if (r != numh)  and  (r != angles.shape[0]):
        print('WARNING: matrix is not full rank')

    mtx = np.linalg.pinv(imtx)

    return mtx


def rank(A, tol=-1):

    S,V,D = np.linalg.svd(A)
    if tol==-1:
        m = np.max(V,axis=0)
        tol = np.multiply(np.max(A.shape), np.spacing(m))

    r = np.sum(V > tol)

    return r


def rcosFn(width=1,position=0,values=[0,1]):
    """

    Args:
        width:
        position:
        values:

    Returns:
        X:
        Y:
    """


    sz = 256  # arbitrary!

    X    = np.pi * np.array([i for i in range(-sz-1,2)]) / (2*sz)

    Y = values[0] + (values[1] - values[0]) * np.cos(X)**2

    #  Make sure end values are repeated, for extrapolation...
    Y[0] = Y[1]
    Y[sz+2] = Y[sz+1]

    X = position + (2*width/np.pi) * (X + np.pi/4)

    return [X,Y]



def pointOp(im, lut, origin, increment):

    #function res = pointOp(im, lut, origin, increment, warnings)
    # NOTE: THIS CODE IS NOT ACTUALLY USED! (MEX FILE IS CALLED INSTEAD in matlab)
    #im = im.T # instead of ,order='F' in reshape below

    X = origin + increment * np.array([i for i in range(0,lut.size)])
    Y = lut.flatten()

    res = np.reshape(np.interp(im.flatten(), X, Y), im.shape) # !!! define interp1 function

    return res
