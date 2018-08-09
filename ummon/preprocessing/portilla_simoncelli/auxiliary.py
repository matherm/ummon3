from __future__ import division

import numpy as np
from numpy.polynomial.polynomial import polyval
from scipy.io import loadmat

from .config import  FULLY_COVERED


def forceAutoCorr(img, autoCorro, p=1):
    """
    Forces the autocorrelation autoCorr to an image img by convolving it with
    an even filter of len(autoCorr) in such a way that the image cotains change
    as less as possible.
    
    Out nimg, snr, Chf]
    nimg: new image
    snr: signal noise ratio - distance to img before ?!?
    Chf: Fourier tranform of the filter that forces the autocorrelation
    """
    # Compute the autocorrelation function of the original image
    [Ny, Nx] = img.shape
    Nc = len(autoCorro)  # normally Nc<<Nx

    imgFft = np.fft.fft2(img.copy())
    imgFft2 = np.absolute(imgFft) ** 2
    # autoCorr of input image
    corrBefore = np.divide(np.fft.fftshift(np.real(np.fft.ifft2(imgFft2.copy()))), (2 - int(np.isrealobj(img))))
    # ac = np.fft.fftshift(np.real(np.fft.ifft2(np.absolute(np.fft.fft2(lpResidual))**2)))/(len(real[-1][0])**2)

    autoCorro = autoCorro * img.size  # Unnormalize the previously normalized correlation

    # center of img
    cy = Ny / 2 + 1
    cx = Nx / 2 + 1
    # y surrounding of autoCorr (9,9) in y direction from center is Lc=4
    Lc = int((Nc - 1) / 2)  ## (Nc-1)/2

    # desired correlation
    corrAfter = autoCorro.copy()
    # p: rate of change ?
    autoCorro = p * autoCorro + (1 - p) * corrBefore[int(cy - Lc - 1):int(cy + Lc),int(cx - Lc - 1):int(cx + Lc)]

    # Compare the actual correlation with the desired one
    u = np.divide( np.sum(corrAfter ** 2), np.sum((corrAfter - corrBefore[int(cy-Lc-1):int(cy+Lc), int(cx-Lc-1):int(cx+Lc)])**2))
    snrV = 10 * np.log10(u)
    # Take just the part that has influence on the samples of Cy (Cy=conv(Cx,Ch))
    corrBefore = corrBefore[int(cy-2*Lc-1):int(cy+2*Lc), int(cx-2*Lc-1):int(cx+2*Lc)]
    # Build the matrix that performs the convolution Cy1=Tcx*Ch1
    Ncx = (4 * Lc + 1)
    M = int((Nc ** 2 + 1) / 2)
    Tcx = np.zeros((M, M))

    for i in range(Lc+1, 2*Lc+1):
        for j in range(Lc+1, 3*Lc+2):
            nm = (i - Lc - 1) * (2 * Lc + 1) + j - Lc
            ccx = np.copy(corrBefore[i-Lc-1:i+Lc, j-Lc-1:j+Lc])
            ccxi = ccx[::-1,::-1]  # np.copy(np.rot90(np.rot90(ccx))) #ccx[::-1,::-1]
            ccx = ccx + ccxi
            ccx[Lc, Lc] = np.divide(ccx[Lc, Lc], 2)
            ccx = ccx.conj().T.flatten(order='F')
            Tcx[nm-1,:] = ccx[0:M] # .conj().T

    i = 2 * Lc + 1
    for j in range(Lc + 1, 2 * Lc + 2):
        nm = (i - Lc - 1) * (2 * Lc + 1) + j - Lc
        ccx = np.copy(corrBefore[i - Lc - 1:i + Lc, j - Lc - 1:j + Lc])
        ccxi = np.copy(ccx[::-1,::-1])
        ccx = ccx + ccxi
        ccx[Lc, Lc] = np.divide(ccx[Lc, Lc], 2)
        ccx = ccx.conj().T.flatten(order='F')
        Tcx[nm-1,:] = ccx[0:M]#.conj().T

    # Rearrange autoCorr indices and solve the equation
    autoCorr1 = autoCorro.conj().T.flatten(order='F')
    autoCorr1 = autoCorr1[0:M]

    Ch1 = np.dot(np.linalg.pinv(Tcx), autoCorr1) # !! usually inv instead of pinv

    # Rearrange Ch1
    # B = np.copy(np.rot90(np.rot90(Ch1)))
    # Ch1 = np.array([[Ch1], [Ch1[-2:-1:-1]]])
    Ch1 = np.concatenate((Ch1, Ch1[-2::-1]), axis=0)  # Ch1.shape[0]-2::-1
    Ch = np.reshape(Ch1, (Nc, Nc),order='F').conj().T

    # Compute Y as conv(X,H) in the Fourier domain
    aux = np.zeros((Ny, Nx))
    aux[int(cy-Lc-1):int(cy+Lc), int(cx-Lc-1):int(cx+Lc)] = Ch
    Ch = np.fft.fftshift(aux)
    Chf = np.real(np.fft.fft2(Ch))

    nimgFft = np.multiply(imgFft.real, np.sqrt(np.absolute(Chf.copy())))
    nimg = np.fft.ifft2(nimgFft.copy()).real

    return [nimg, snrV, Chf]


def adjustCorr1s(img, Co, mode, p=1):
    """
    MODE:
    0 => choose randomly from the space of linear solutions
    1 => simplest soln
    2 => minimize angle change (DEFAULT)
    3 => SVD minimal vector change soln
    """
    #
    dim3 = False
    if len(img.shape)==3:
        dim3 = True
        [Ny, Nx, Nclr] = img.shape
        img = np.reshape(img, (Ny*Nx, Nclr))

    C = np.dot(np.transpose(img), img) / img.shape[0]  # computation of inner product !!!

    [eigValues, eigVectors] = np.linalg.eig(C)  # D: eigenvalues, E: eigenvectors
    idx = eigValues.argsort()
    eigValues = eigValues[idx]  # sorted eigValues - increasing
    eigValuesMatrix = np.zeros((eigVectors.shape), dtype=np.complex)
    np.fill_diagonal(eigValuesMatrix, np.sqrt(eigValues[::-1].astype(np.complex)))
    eigVectors[:,::-1] = eigVectors[:,idx]

    Co0 = Co
    Co = (1 - p) * C + p * Co

    [eigValues0, eigVectors0] = np.linalg.eig(Co)
    idx = eigValues0.argsort()
    eigValues0 = eigValues0[idx]
    eigValuesMatrix0 = np.zeros((eigVectors0.shape), dtype=np.complex)
    np.fill_diagonal(eigValuesMatrix0, np.sqrt(eigValues0[::-1].astype(np.complex)))
    eigVectors0[:,::-1] = eigVectors0[:,idx]

    # depends on mode
    if mode == 0:
        print("MODE 0 not implemented yet.")
    elif mode == 1:
        print("MODE 1 not implemented yet.")
    elif mode == 2:
        Orth = np.dot(eigVectors.conj().T, eigVectors0)
    else:
        [U, S, V] = np.linalg.svd(np.dot(np.dot(np.dot(eigValuesMatrix, eigVectors.conj().T), eigVectors0),
                                         np.inverse(eigValuesMatrix0)), full_matrices=True)
        Orth = np.dot(U, V.conj().T)

    M = np.dot(np.dot(np.dot(np.dot(eigVectors, np.linalg.pinv(eigValuesMatrix)), Orth), eigValuesMatrix0), # usually np.linalg.inv
               eigVectors0.conj().T)


    nimg = np.dot(img, M)

    if dim3:
        nimg = np.reshape(nimg, (Ny,Nx,Nclr))
        #img = img[0]

    snr1 = 10 * np.log10(np.sum(Co0 ** 2) / np.sum((Co0 - C) ** 2))

    return [nimg, snr1, M]


def adjustCorr2s(X, Cx, Y, Cxy, mode, p):
    """
    Linearly adjust variables in X to have correlation Cx, and cross-correlation Cxy.
    Rows of X, Y, and newX are samples of (random) row-vectors, such that:
    1:  newX = X * Mx + Y * My
    2:  newX' * newX = Cx
    3:  newX' * Y = Cxy

    MODE is optional:
    0 => choose randomly from the space of linear solutions
    1 => simplest soln
    2 => minimize angle change
    3 => Simple rotational (DEFAULT) 
    4 => SVD minimal vector change soln
    """
    Bx = np.dot(X.conj().T, X) / X.shape[0]  # inner product !!!
    Bxy = np.dot(X.conj().T, Y) / X.shape[0]
    By = np.dot(Y.conj().T, Y) / X.shape[0]  # inner product !!!

    # if(By.size == 16):

    iBy = np.linalg.pinv(By) # pinv instead of inv to avoid singular matrix in defect synth!!!
    #iBy = np.linalg.inv(By)
    #eye = np.eye(len(By), dtype=int)
    #iBy = np.linalg.lstsq(By, eye)
    #iBy = iBy[0]

    # else:
    #    iBy = np.linalg.pinv(By) # if By is (4,4) matrix --> inv()

    Current = Bx - (np.dot(np.dot(Bxy, iBy), Bxy.conj().T))
    Cx0 = Cx
    Cx = (1 - p) * Bx + p * Cx
    Cxy0 = Cxy
    Cxy = (1 - p) * Bxy + p * Cxy
    Desired = Cx - (np.dot(np.dot(Cxy, iBy), Cxy.conj().T))

    [eigValues, eigVectors] = np.linalg.eig(Current)  # D: eigenvalues, E: eigenvectors
    idx = eigValues.argsort()
    eigValues = eigValues[idx]
    eigValuesMatrix = np.zeros((eigVectors.shape), dtype=np.complex)
    np.fill_diagonal(eigValuesMatrix, np.sqrt(eigValues[::-1].astype(np.complex)))
    eigVectors[:,::-1] = eigVectors[:,idx]

    [eigValues0, eigVectors0] = np.linalg.eig(Desired)
    # Ind = np.argsort(Do,axis=None)
    idx = eigValues0.argsort()
    eigValues0 = eigValues0[idx]
    eigValuesMatrix0 = np.zeros((eigVectors0.shape), dtype=np.complex)
    np.fill_diagonal(eigValuesMatrix0, np.sqrt(eigValues0[::-1].astype(np.complex)))
    eigVectors0[:,::-1] = eigVectors0[:,idx]


    # depends on mode - take simple mode
    if mode == 0:
        print("MODE 0 is not implemented yet.")
    elif mode == 1:
        Orth = np.eye(eigValuesMatrix.shape, k=1)
    elif mode == 2:
        A = np.array([np.eye(Cx.shape, k=1), np.dot(-iBy, Bxy.conj().T)])
        Ao = np.array([np.eye(Cx.shape, k=1), np.dot(-iBy, Cxy.conj().T)])
        [U, S, V] = np.linalg.svd(np.dot(np.dot(np.dot(np.dot(eigVectors.conj().T, np.pinv(A)), Ao), eigVectors0),
                                         np.inv(eigValuesMatrix0)), full_matrices=True)
        Orth = np.dot(U, V.conj().T)
    elif mode == 3:
        Orth = np.dot(eigVectors.conj().T, eigVectors0)
    else:
        print("MODE >3 not implemented yet.")

    Mx = np.dot(np.dot(np.dot(np.dot(eigVectors, np.linalg.pinv(eigValuesMatrix)), Orth), eigValuesMatrix0), # !!! to avoid singular matrix in defect synth !!!
                eigVectors0.conj().T)
    My = np.dot(iBy, (Cxy.conj().T - np.dot(Bxy.conj().T, Mx)))
    newX = np.dot(X, Mx) + np.dot(Y, My)

    # print("Cx0.shape: ", Cx0.shape) #()
    # print("Cx0.dtype: ", Cx0.dtype)
    # print("Cx0: ", Cx0)
    # print("Cx0.size: ", Cx0.size)
    # print("Bx.shape: ", Bx.shape) # (1,1)
    # print("Bx.dtype: ", Bx.dtype)
    # print("Bx: ", Bx)
    # print("Bx.size: ", Bx.size)

    if Cx0.size == 1 and Bx.size == 1:
        if Bx == Cx0:
            snr1 = np.inf
        else:
            h1 = Cx0 ** 2
            h2 = (Cx0 - Bx) ** 2
            snr1 = 10 * np.log10(np.sum(h1) / np.sum(h2))

    else:
        if len(Cx0.shape) == 1:
            Cx0 = np.atleast_2d(Cx0)

        mask = Cx0 != Bx
        # print("mask:", mask)
        if mask[mask == True].size == 0:
            snr1 = np.inf
        else:
            h1 = np.zeros((mask.shape))
            h2 = np.zeros((mask.shape))
            h1[mask] = Cx0[mask] ** 2
            h2[mask] = (Cx0[mask] - Bx[mask]) ** 2
            snr1 = 10 * np.log10(np.sum(h1) / np.sum(h2))

    # print("Cxy0.shape: ", Cxy0.shape)  # (8,)
    # print("Cxy0.dtype: ", Cxy0.dtype)
    # print("Cxy0: ", Cxy0)
    # print("Cxy0.size: ", Cxy0.size)
    # print("Bxy.shape: ", Bxy.shape) # (1,8)
    # print("Bxy.dtype: ", Bxy.dtype)
    # print("Bxy: ", Bxy)
    # print("Bxy.size: ", Bxy.size)

    if Cxy0.size == 1 and Bxy.size == 1:
        if Bxy == Cxy0:
            snr2 = np.inf
        else:
            h1 = Cxy0 ** 2
            h2 = (Cxy0 - Bxy) ** 2
            snr1 = 10 * np.log10(np.sum(h1) / np.sum(h2))

    else:
        if len(Cxy0.shape) == 1:
            Cxy0 = np.atleast_2d(Cxy0)

        mask = Cxy0 != Bxy
        # print("mask:", mask)
        if mask[mask == True].size == 0:
            snr2 = np.inf
        else:
            h1 = np.zeros((mask.shape))
            h2 = np.zeros((mask.shape))
            h1[mask] = Cxy0[mask] ** 2
            h2[mask] = (Cxy0[mask] - Bxy[mask]) ** 2
            snr2 = 10 * np.log10(np.sum(h1) / np.sum(h2))

    # if Cxy0.size == 1 and Bxy.size == 1: #dimension == 0 or 1
    #    print("2. SIZE IS == 1")
    #    h1 = Cxy0**2
    #    h2 = Cxy0-Bxy**2
    #    snr2 = 10*np.log10(np.sum(h1)/np.sum(h2))

    # else:
    #    if len(Cxy0.shape) == 1:
    #        Cxy0 = np.atleast_2d(Cxy0)
    #    
    #    mask = Cxy0!=Bxy
    #    #print("mask:", mask)
    #    
    #    if mask[mask==True].size == 0:
    #        snr2 = np.inf
    #    else:
    #        h1 = np.zeros((mask.shape))
    #        h2 = np.zeros((mask.shape))
    #        h1[mask] = Cxy0[mask]**2
    #        h2[mask] = Cxy0[mask]-Bxy[mask]**2
    #        snr2 = 10*np.log10(np.sum(h1)/np.sum(h2))

    #print("snr1", snr1)
    #print("snr2", snr2)

    return [newX, snr1, snr2, Mx, My]


def signaltonoise(s, n):
    """
    This function computes the signal-to-noise ratio in dB. (But doesn't subtract means.)
    """
    if np.array(s).size == 1:
        es = np.absolute(s) ** 2
    else:
        es = sum(sum(np.absolute(s) ** 2))

    if n.size == 1:
        en = np.absolute(n) ** 2
    else:
        en = sum(sum(np.absolute(n) ** 2))

    X = 10 * np.log10(np.divide(es, en))

    return X


def modskew(ch, sk, p):
    """
    This function adjust the skewness sk to an image ch using gradient projection. 
    """
    N = ch.size
    me = np.mean(ch)
    ch = ch - me
    m = np.zeros((6))
    a = np.zeros((7))
    # m[0] = me

    for n in range(1, 6):
        m[n] = np.mean(ch**(n + 1))

    #print("moment 2:", m[1])
    #print("moment 3:", m[2])
    sd = np.sqrt(np.complex(m[1])).real  # standard deviation #!!! complex or absolute to avoid nan
    s = np.divide(m[2], sd**3)  # original skewness
    # snrk = scipy.stats.signaltonoise(sk, sk-s)
    snrk = signaltonoise(sk, sk - s)
    sk = np.multiply(s, (1 - p)) + np.multiply(sk, p)

    #print("sd: ", sd)
    #print("s: ", s)
    #print("snrk: ", snrk)
    #print("sk: ", sk)
    # Define the coefficients of the numerator (A*lam^3+B*lam^2+C*lam+D)
    A = m[5] - 3 * sd * s * m[4] + 3 * sd ** 2 * (s ** 2 - 1) * m[3] + sd ** 6 * (2 + 3 * s ** 2 - s ** 4)
    B = 3 * (m[4] - 2 * sd * s * m[3] + sd ** 5 * s ** 3)
    C = 3 * (m[3] - sd ** 4 * (1 + s ** 2))
    D = s * sd ** 3
    # A = m[5] - np.multiply(np.multiply(np.multiply(3,sd),s),m[4]) + np.multiply(np.multiply(np.multiply(3,sd**2),(s**2 - 1)),m[3]) + np.multiply(sd**6,(2 + np.multiply(3,s**2) - s**4))
    # B = np.multiply(3,(m[4] - np.multiply(np.multiply(np.multiply(2,sd),s),m[3]) + sd**5 * s**3))
    # C = np.multiply(3,(m[3] - np.multiply(sd**4,(1 + s**2))))
    # D = np.multiply(s,sd**3)

    a[6] = A ** 2
    a[5] = 2 * A * B
    a[4] = B ** 2 + 2 * A * C
    a[3] = 2 * (A * D + B * C)
    a[2] = C ** 2 + 2 * B * D
    a[1] = 2 * C * D
    a[0] = D ** 2
    # a[6] = A**2
    # a[5] = np.multiply(np.multiply(2,A),B)
    # a[4] = B**2+np.multiply(np.multiply(2,A),C)
    # a[3] = np.multiply(2,(np.multiply(A,D)+np.multiply(B,C)))
    # a[2] = C**2+np.multiply(np.multiply(2,B),D)
    # a[1] = np.multiply(np.multiply(2,C),D)
    # a[0] = D**2

    # Define the coefficients of the denominator (A2+B2*lam^2)

    A2 = sd ** 2
    B2 = m[3] - (1 + s**2) * sd**4
    # A2 = sd**2
    # B2 = m[4] - np.multiply((1+s**2), sd**4)

    b = np.zeros((7))
    b[6] = B2 ** 3
    b[4] = 3 * A2 * B2 ** 2
    b[2] = 3 * A2 ** 2 * B2
    b[0] = A2 ** 3
    # b = np.zeros((7))
    # b[6] = B2**3
    # b[4] = np.multiply(np.multiply(3,A2),B2**2)
    # b[2] = np.multiply(np.multiply(3,A2**2),B2)
    # b[0] = A2**3

    # Now compute its derivative with respect to lambda
    d = np.zeros((8))
    d[7] = B * b[6]
    d[6] = 2 * C * b[6] - A * b[4]
    d[5] = 3 * D * b[6]
    d[4] = C * b[4] - 2 * A * b[2]
    d[3] = 2 * D * b[4] - B * b[2]
    d[2] = -3 * A * b[0]
    d[1] = D * b[2] - 2 * B * b[0]
    d[0] = -C * b[0]
    # d = np.zeros((8))
    # d[7] = np.multiply(B,b[6])
    # d[6] = np.multiply(np.multiply(2,C),b[6]) - np.multiply(A,b[4])
    # d[5] = np.multiply(np.multiply(3,D),b[6])
    # d[4] = np.multiply(C,b[4]) - np.multiply(np.multiply(2,A),b[2])
    # d[3] = np.multiply(np.multiply(2,D),b[4]) - np.multiply(B,b[2])
    # d[2] = np.multiply(np.multiply(-3, A), b[0])
    # d[1] = np.multiply(D, b[2]) - np.multiply(np.multiply(2,B),b[0])
    # d[0] = np.multiply(-C, b[0])

    d = d[7::-1]

    # inf * 0 = nan /!\
    #if d[np.isnan(d)].size > 0 or d[np.isinf(d)].size > 0:
    #    print("d.isnan count: ", d[np.isnan(d)].size)
    #    print("d.isinf count: ", d[np.isinf(d)].size)

    mMlambda = np.roots(d)
    # mMlambda = np.polynomial.polynomial.polyroots(d)
    # mMlambda = mMlambda[::-1]

    tg = np.divide(np.imag(mMlambda), np.real(mMlambda))

    # mMlambda = np.real(mMlambda[find(np.absolute(tg)<1e-6)])
    mMlambda = np.real(mMlambda[(np.absolute(tg) < 1e-6)])
    # mMlambda = np.real(mMlambda[i for i in range(len(tg)) if np.absolute(tg[i])<1e-6 ])

    # lNeg = mMlambda[find(mMlambda<0)]
    lNeg = mMlambda[(mMlambda < 0)]
    # lNeg = mMlambda[i for i in range(len(mMlambda)) if mMlambda[i]<0]

    if len(lNeg) == 0:
        lNeg = np.array([-1 / np.spacing(1)])

    # lPos = mMlambda[find(mMlambda>=0)]
    lPos = mMlambda[(mMlambda >= 0)]
    # lPos = mMlambda[i for i in range(len(mMlambda)) if mMlambda[i]>=0]

    if len(lPos) == 0:
        lPos = np.array([1 / np.spacing(1)])

    #lmi = max(lNeg.tolist())
    #lma = min(lPos.tolist())

    #lam = [lmi, lma]
    #mMnewSt = np.sqrt(np.divide(polyval(lam, [D, C, B, A]), (polyval(lam, b)))) #** 0.5  # order? power before division ?
    #skmin = min(mMnewSt)
    #skmax = max(mMnewSt)

    # Given a desired skewness, solves for lambda
    # The equation is sum(c.*lam.^(0:6))=0

    c = a - b * sk ** 2

    c = c[::-1]
    # print("c: ", c)
    r = np.roots(c)

    # Chose the real solution with minimum absolute value with the right sign
    lam = np.zeros((6))
    for n in range(6):
        tg = np.divide(np.imag(r[n]), np.real(r[n]))
        if (np.absolute(tg) < 1e-6) and (np.sign(np.real(r[n])) == np.sign(sk - s)):
            #print("ANY VALUE IS SET IN LAM!!")
            lam[n] = np.real(r[n])
        # elif (np.absolute(tg)<1e-6) and (np.sign(-np.real(r[n]))==np.sign(sk-s)):
        #    print("NEGATIVE CORRELATION")
        #    lam[n] = -np.real(r[n])
        else:
            lam[n] = np.NINF

    p = [D, C, B, A]

    mask = np.invert(np.isinf(lam))
    lam = lam[mask]  # reduce to not infinite elements in lam

    # if no value is set in lam:
    if lam.size == 0:  # and Warn: !!!
        print('Warning: Skew adjustment skipped!')
        lam = np.array([0])  # to avoid below failure - broadcast together with shapes (0,) (32,32)

    if len(lam) > 1:
        #print("CASE 1")
        foo = np.sign(polyval(lam, p))
        if np.any(foo == 0):
            #print("CASE 1-1")
            lam = lam[(foo == 0)]
        else:
            #print("CASE 1-2")
            lam = lam[(foo == np.sign(sk))]  ## rejects the symmetric solution ?!?

        if len(lam) > 0:
            #print("CASE 1-3")
            lam = lam[(np.absolute(lam) == min(np.absolute(lam)))]  # the smallest that fix the skew
            lam = lam[0]
        else:
            #print("CASE 1-4")
            lam = np.array([0])  # to avoid below failure - broadcast together with shapes (0,) (32,32)
    #else:
    #    print("CASE 2")

    # Modify the channel
    chm = ch + lam * (ch ** 2 - sd ** 2 - sd * s * ch)  # adjust the skewness
    chm = chm * np.sqrt(np.complex(np.divide(m[1], np.mean(chm**2)))).real # adjust the variance # from now on complex
    chm = chm + me  # adjust the mean

    return [chm, snrk]


def modkurt(ch, k, p):
    """
    This function modifies the kurtosis in one step by moving in
    gradient direction until reaching the desired kurtosis value.
    """
    #ch = ch.real # to avoid the complex input which comes from modskew output
    me = np.mean(ch)
    ch = ch - me

    # Compute the moments
    m = np.zeros((12))

    for n in range(1, 12):
        m[n] = np.mean(ch ** (n + 1))  # np.nanmean()

    # TESTING - second moment is 0:
    # m[1] = 0

    # The original kurtosis
    k0 = np.divide(m[3], m[1] ** 2)  # force inf value for zero division !!!!
    snrk = signaltonoise(k, k - k0)  # if k-k0 is negative infinite -->
    if snrk > 60:
        chm = ch + me
        return [chm, snrk]

    k = np.multiply(k0, (1 - p)) + k * p  # In TEST CASE: Inf*0 = line 422 fails: Nan --> mMlambda = np.roots(d)

    # Some auxiliar variables
    a = np.divide(m[3], m[1])  # force inf value for zero division !!!!

    # Coeficients of the numerator (A*lam^4+B*lam^3+C*lam^2+D*lam+E)

    A = m[11] - 4 * a * m[9] - 4 * m[2] * m[8] + 6 * a ** 2 * m[7] + 12 * a * m[2] * m[6] + 6 * m[2] ** 2 * m[5] - \
        4 * a ** 3 * m[5] - 12 * a ** 2 * m[2] * m[4] + a ** 4 * m[3] - 12 * a * m[2] ** 2 * m[3] + \
        4 * a ** 3 * m[2] ** 2 + 6 * a ** 2 * m[2] ** 2 * m[1] - 3 * m[2] ** 4
    B = 4 * (m[9] - 3 * a * m[7] - 3 * m[2] * m[6] + 3 * a ** 2 * m[5] + 6 * a * m[2] * m[4] + 3 * m[2] ** 2 * m[3] - \
             a ** 3 * m[3] - 3 * a ** 2 * m[2] ** 2 - 3 * m[3] * m[2] ** 2)
    C = 6 * (m[7] - 2 * a * m[5] - 2 * m[2] * m[4] + a ** 2 * m[3] + 2 * a * m[2] ** 2 + m[2] ** 2 * m[1])
    D = 4 * (m[5] - a ** 2 * m[1] - m[2] ** 2)
    E = m[3]

    # Define the coefficients of the denominator (F*lam^2+G)^2

    F = D / 4
    G = m[1]

    # Now I compute its derivative with respect to lambda
    # (only the roots of derivative = 0 )
    d = np.zeros((5))

    d[0] = B * F
    d[1] = 2 * C * F - 4 * A * G
    d[2] = 4 * F * D - 3 * B * G - D * F
    d[3] = 4 * F * E - 2 * C * G
    d[4] = -D * G

    mMlambda = np.roots(d)

    tg = np.imag(mMlambda) / np.real(mMlambda)
    mMlambda = mMlambda[(abs(tg) < 1e-6)]
    lNeg = mMlambda[(mMlambda < 0)]
    if len(lNeg) == 0:
        lNeg = np.array([-1 / np.spacing(1)]) # !!! np.array([], dtype='np.float64') np.finfo(np.float32).eps

    lPos = mMlambda[(mMlambda >= 0)]
    if len(lPos) == 0:
        lPos = np.array([1 / np.spacing(1)])

    lmi = max(lNeg)
    lma = min(lPos)

    #lam = [lmi, lma]
    #mMnewKt = np.divide(polyval(lam, [E, D, C, B, A]), (polyval(lam, [G, 0, F]))) ** 2
    #kmin = min(mMnewKt)
    #kmax = max(mMnewKt)

    # Given a desired kurtosis, solves for lambda

    # Coeficients of the algebraic equation

    c0 = E - k * G ** 2
    c1 = D
    c2 = C - 2 * k * F * G
    c3 = B
    c4 = A - k * F**2

    # Solves the equation
    r = np.roots(np.array([c4, c3, c2, c1, c0]))

    # Chose the real solution with minimum absolute value with the rigth sign

    tg = np.imag(r) / np.real(r)
    # lambda = real(r(find(abs(tg)<1e-6)));
    lambda0 = np.real(r[(abs(tg) == 0)])
    if len(lambda0) > 0:
        lam = lambda0[(np.absolute(lambda0) == min(np.absolute(lambda0)))]
        lam = lam[0]
    else:
        lam = np.array([0])

    # Modify the channel

    chm = ch + lam * (ch ** 3 - a * ch - m[2])  # adjust the kurtosis
    chm = chm * np.sqrt(np.complex(np.divide(m[1], np.mean(chm ** 2)))).real  # adjust the variance
    chm = chm + me  # adjust the mean

    return [chm, snrk]


def mkAngle(sz, phase, origin):
    """
    This function modifies the kurtosis in one step by moving in
    gradient direction until reaching the desired kurtosis value.
    """
    # xv, yv = meshgrid(x, y, sparse=False, indexing='ij')
    [xramp, yramp] = np.meshgrid(np.array([i for i in range(1,sz[1]+1)]) - origin[1], np.array([i for i in range(1,sz[0]+1)]) - origin[0])

    #res = np.arctan2(xramp,yramp)
    res = np.arctan2(yramp, xramp)

    res = np.mod(res + (np.pi - phase), 2 * np.pi) - np.pi

    return res


def expand(img, factor):
    """
    Expand spatially an image img in a factor
    in X and in Y.
    img may be complex.
    It fills in with zeros in the Fourier domain.

    img_expanded = expand(img, factor)

    """
    [my, mx] = img.shape
    my = factor * my
    mx = factor * mx
    Expand = np.zeros((my,mx),dtype=np.complex)
    ImgFft = factor**2 * np.fft.fftshift(np.fft.fft2(img.copy()))
    y1 = int(my / 2 + 2 - my / (2*factor))
    y2 = int(my / 2 + my / (2*factor))
    x1 = int(mx / 2 + 2 - mx / (2*factor))
    x2 = int(mx / 2 + mx / (2*factor))
    Expand[y1-1:y2,x1-1:x2] = ImgFft[1:int(my/factor),1:int(mx/factor)]
    Expand[y1-2,x1-1:x2] = ImgFft[0,1:int(mx/factor)]/2
    Expand[y2,x1-1:x2] = ((ImgFft[0,int(mx/factor-1):0:-1]/2).conj().T).T #.conj().T
    Expand[y1-1:y2,x1-2] = ImgFft[1:int(my/factor),0]/2
    Expand[y1-1:y2,x2] = ((ImgFft[int(my/factor):0:-1,0]/2).conj().T).T #.conj().T
    esq = ImgFft[0,0]/4
    Expand[y1-2,x1-2] = esq
    Expand[y1-2,x2] = esq
    Expand[y2,x1-2] = esq
    Expand[y2,x2] = esq
    Expand = np.fft.fftshift(Expand.copy())
    img_expanded = np.fft.ifft2(Expand.copy())
    if (img.imag == 0).all:
        img_expanded = img_expanded.real

    return img_expanded

def shrink(t, f):
    # function ts=shrink(t,f)

    # It shrinks an image in a factor f
    # in each dimension.
    #	ts = shrink(t,f)
    # ts may also be complex.
    # See also: expand.m, blurDn.m

    [my,mx] = t.shape
    T = np.fft.fftshift(np.fft.fft2(t)) / f**2
    b = mx/f
    if b == 0:
        b = 1
    Ts = np.zeros((my/f, b))
    y1 = my / 2 + 2 - my / (2*f)
    y2 = my / 2 + my / (2*f)
    x1 = mx / 2 + 2 - mx / (2*f)
    x2 = mx / 2 + mx / (2*f)
    Ts[1:my/f,1:mx/f] = T[y1-1:y2,x1-1:x2]
    Ts[0,1:mx/f] = (T[y1-2,x1-1:x2] + T[y2,x1-1:x2])/2
    Ts[1:my/f,0] = (T[y1-1:y2,x1-2] + T[y1-1:y2,x2])/2
    Ts[0,0] = (T[y1-2,x1-2] + T[y1-2,x2] + T[y2,x1-2] + T[y2,x2])/4
    Ts = np.fft.fftshift(Ts)
    ts = np.fft.ifft2(Ts)
    if (t.imag==0).all():
        ts = ts.real

    return ts


def loadFromMat(variable, deep=True, color=False):
    # matlab computed statistics:
    if color:
        matparams = loadmat("../matlab/colorTextureSynth/"+str(variable)+".mat")[variable]
    else:
        matparams = loadmat("../matlab/textureSynth/"+str(variable)+".mat")[variable]
    if deep:
        matdict = matparams[0,0]
    else:
        matdict = matparams

    return matdict

def scale(matrix):
    # scale min and max value onto 0 and 1
    min = np.max(matrix)
    matrix_scaled = matrix.copy() - min
    max = np.max(matrix_scaled)
    matrix_scaled = np.divide(matrix_scaled, max)
    return matrix_scaled

def scale2(matrix, matrix2):
    """
    """
    # norm two matrices betewen the min of both and max of both matrices
    min = np.minimum(np.min(matrix), np.min(matrix2))

    shifted = matrix - min
    shifted2 = matrix2 - min

    max = np.maximum(np.max(shifted),np.max(shifted2))

    normed = shifted/max
    normed2 = shifted2/max


    return normed, normed2

def norm(img):
    # norm img values from 0 - 255 in-between 0 - 1
    if (img<0).any() or (img>255).any():
        if (img>(-1)).all() and (img<2).all():
            print('not normed yet')
            img[img<0] = 0
            img[img>1] = 1
            return img
        raise NameError('Img can not be normed because its range is out of 0-255.')
    elif (img>=0).all() and (img<=1).all():
        # already normed in-between 0 - 1
        return img

    if (img>1).any():
        if (img<2).all():
            print('not cut of yet')
            img[img<0] = 0
            img[img>1] = 1
            return img
        else:
            img_normed = np.divide(img.copy(), 255)


    return img_normed

def random_noise(mean, var, shape):

    if mean<0 or mean>1:
        raise NameError('Random noise requires normed values. Given mean value is not inbetween range: 0 - 1.')

    noise = np.random.normal(mean, np.sqrt(var), size=shape)
    #noise = np.random.normal(1, np.sqrt(1), size=shape)
    #noise = ip.scale_vals(noise,0,1)

    # equal distributed
    #noise = np.divide(np.random.randint(0,255,size=shape).astype(np.float64), 255)

    if FULLY_COVERED:
        # (A) border (values on 0 and 1 can be greater than distribution provide)
        noise[noise<0] = 0
        noise[noise>1] = 1
    else:
        # (B) scale in-between 0 - 1 (standard deviation can be distorted)
        noise = ip.scale_vals(noise.astype(np.float64), 0, 1) #norm?
        #noise = scale(noise.astype(np.float64)) #norm?
    #print('Initial mean: ', mean)
    #print('Initial var', var)
    #print('Random mean: ', np.mean(noise))
    #print('Random var', np.var(noise))
    #print('Random min: ', np.min(noise))
    #print('Random max', np.max(noise))

    #ip.save_img(noise, 'images/random/generated_noise.png')

    return noise


def norm_measure(a,b,measure):
    ret = np.divide(measure, np.sqrt(np.multiply(np.sum(a**2),np.sum(b**2))))
    return ret


def square_diff(a,b,normed=False):

    # ToDo: norm between 0 - 1
    if normed:
        return np.sqrt((a-b)**2) / 2 # divide to avoid difference 2
    else:
        return (a-b) # np.divide( np.sum((a-b)**2), np.sum((a+b)**2) )

# def correlation(a,b,normed=False):
#     ret = np.sum(np.multiply(a,b)**2)
#     if normed:
#         ret = norm_measure(a,b,ret)
#     # to get percentual result divide by maximal result
#     if type(a) == np.ndarray:
#         pixels = np.prod(a.shape)
#     else:
#         pixels = 1
#     ret = np.divide(ret, pixels * (255*255)**2)
#     return ret
#
# def covariance(a,b,normed=False):
#     ret = np.sum(np.multiply(a-np.mean(a),b-np.mean(b))**2)
#     if normed:
#         ret = norm_measure(a,b,ret)
#     # to get percentual result divide by maximal result
#     if type(a) == np.ndarray:
#         pixels = np.prod(a.shape)
#     elif type(a)==int or type(a)==np.int:
#         pixels = 1
#     else:
#         pixels = len(a)
#     ret = np.divide(ret, pixels * (255*255)**2)
#     return ret


def compare(a, b, string=0, normed=False, all=False, measure='all_close', rtol=1.e-5, atol=1.e-8):
    """
    measure: 'all_close', 'percentual_close', 'square_diff', 'correlation', 'corr_coeff'
    """

    a = np.atleast_2d(np.array(a))
    b = np.atleast_2d(np.array(b))

    if normed:
        a, b = scale2(a, b)
        #b = norm(b)
    else:
        a = a
        b = b

    # used for matlab comparison
    if measure=='all_close':
        if np.allclose(a, b, rtol, atol):
            print(string, ": do agree [+]")
            return True
        else:
            print(string, ": don't agree [-]")
            return False

    # used to compare parameters: original - defect
    elif measure=='percentual_close':
        agree = np.sum(np.isclose(a, b, rtol, atol)==True)
        not_agree = np.sum(np.isclose(a, b, rtol, atol)==False)
        agreement = agree/(agree+not_agree) * 100

        #print(string," agreement: ", agreement,'%')
        return '{0:.3f}%'.format(agreement)

    elif measure=='is_close':
        return np.isclose(a, b, rtol, atol)

    if measure=='square_diff':
        if normed:
            #ret_max = square_diff(np.full(a.shape, 0), np.full(b.shape, 255))
            ret = square_diff(a, b, normed)
            #ret = np.divide(ret, np.divide(1,ret_max))
        else:
            ret = square_diff(a, b, normed)
        #print(string," sqaure_diff: ",  ret)
        return ret

    # elif measure=='correlation':
    #     ret = correlation(a, b, normed)
    #     print(string," correlation: ",  ret)
    #     return ret
    #
    # elif measure=='covariance':
    #     ret = covariance(a, b, normed)
    #     print(string," covariance: ",  ret)
    #     return ret

    else:
        print("Similarity measure is not defined. Use 'all_close', \
        'percentual_close', 'is_close' or 'square_diff'.")



