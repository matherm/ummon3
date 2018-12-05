import numpy as np

def pyrBandIndices(pInd, band):
    assert 0 <= band <= len(pInd), 'number of bands must be between 0 and number of pyramid bands {}.'.format(
        len(pInd - 1))
    assert pInd.shape[1] == 2, 'indices must be an N x 2 matrix indicating the size of the pyramid subbands'

    ind = 0
    for i in range(band - 1):
        ind += np.prod(pInd[i])
    return np.arange(ind, ind + np.prod(pInd[band - 1])).astype(int)


def pyrBand(pyr, pInd, band):
    if not isinstance(pInd, int):
        pInd = pInd.astype(int)
    if not isinstance(band, int):
        band = int(band)
    return np.reshape(pyr[pyrBandIndices(pInd, band)], (pInd[band - 1, 0], pInd[band - 1, 1]))


def pyrLow(pyr, pInd):
    if not isinstance(pInd, int):
        pInd = pInd.astype(int)
    band = len(pInd)
    if not isinstance(band, int):
        band = int(band)
    return np.reshape(pyr[pyrBandIndices(pInd, band)], pInd[band - 1])


def rCosFn(width=1, position=0, values=(0, 1)):
    assert isinstance(values, tuple), 'values has to be a tuple'

    sz = 256
    x = np.pi * np.arange(-sz - 1, 2) / (2 * sz)
    y = values[0] + (values[1] - values[0]) * (np.cos(x) ** 2)
    y[0] = y[1]; y[sz + 2] = y[sz + 1]

    x = position + (2 * width / np.pi) * (x + np.pi / 4)

    return x, y


def expand(im, fac):
    if not isinstance(im, np.ndarray):
        im = np.array(im)
    assert im.ndim == 2, 'im has to be two dimensional'

    my, mx = im.shape
    my *= fac
    mx *= fac
    imExp = np.zeros((my, mx)).astype(np.complex)
    imFFT = (fac ** 2) * np.fft.fftshift(np.fft.fft2(im))
    yB = int(my / 2 + 2 - my / (2 * fac)) - 1
    yE = int(my / 2 + my / (2 * fac))
    xB = int(mx / 2 + 2 - mx / (2 * fac)) - 1
    xE = int(mx / 2 + mx / (2 * fac))
    imExp[yB: yE, xB: xE] = imFFT[1: int(my / fac), 1 : int(mx / fac)]
    imExp[yB - 1, xB: xE] = imFFT[0, 1: int(mx / fac)] / 2
    imExp[yE, xB: xE] = imFFT[0, int(mx / fac) - 1: 0: -1] / 2
    imExp[yB: yE, xB - 1] = imFFT[1: int(my / fac), 0] / 2
    imExp[yB: yE, xE] = imFFT[int(my / fac) - 1: 0: -1, 0] / 2
    esq = imFFT[0, 0] / 4
    imExp[yB - 1, xB - 1] = esq
    imExp[yB - 1, xE] = esq
    imExp[yE, xB - 1] = esq
    imExp[yE, xE] = esq
    imExpFFT = np.fft.ifft2(np.fft.fftshift(imExp))
    if np.all(im.imag == 0):
        imExpFFT = imExpFFT.real
    return imExpFFT


def pointOp(logRad, y, x):
    return np.reshape(np.interp(logRad.ravel(), x, y), logRad.shape)
