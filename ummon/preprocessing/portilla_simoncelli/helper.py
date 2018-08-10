import numpy as np


def skew(img):
    mn = np.mean(img)
    var = np.var(img)
    return skew2(img, mn, var)


def skew2(img, mn, var):
    if np.isrealobj(img):
        return np.mean((img - mn) ** 3) / (var ** (3 / 2))
    else:
        return complex(np.mean(np.real(img - mn) ** 3) / np.real(var) ** (3 / 2),
                       np.mean(np.imag(img - mn) ** 3) / (np.imag(var) ** (3 / 2)))


def kurt(img):
    mn = np.mean(img)
    var = np.var(img)
    return kurt2(img, mn, var)


def kurt2(img, mn, var):
    if np.isrealobj(img):
        return np.mean(np.absolute(img - mn) ** 4) / (var ** 2)
    else:
        return complex(np.mean(np.real(img - mn) ** 4) / np.real(var) ** (3 / 2),
                       np.mean(np.imag(img - mn) ** 4) / (np.imag(var) ** 2))


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
    Expand = np.zeros((my, mx), dtype=np.complex)
    ImgFft = factor ** 2 * np.fft.fftshift(np.fft.fft2(img.copy()))
    y1 = int(my / 2 + 2 - my / (2 * factor))
    y2 = int(my / 2 + my / (2 * factor))
    x1 = int(mx / 2 + 2 - mx / (2 * factor))
    x2 = int(mx / 2 + mx / (2 * factor))
    Expand[y1 - 1:y2, x1 - 1:x2] = ImgFft[1:int(my / factor), 1:int(mx / factor)]
    Expand[y1 - 2, x1 - 1:x2] = ImgFft[0, 1:int(mx / factor)] / 2
    Expand[y2, x1 - 1:x2] = ((ImgFft[0, int(mx / factor - 1):0:-1] / 2).conj().T).T  # .conj().T
    Expand[y1 - 1:y2, x1 - 2] = ImgFft[1:int(my / factor), 0] / 2
    Expand[y1 - 1:y2, x2] = ((ImgFft[int(my / factor):0:-1, 0] / 2).conj().T).T  # .conj().T
    esq = ImgFft[0, 0] / 4
    Expand[y1 - 2, x1 - 2] = esq
    Expand[y1 - 2, x2] = esq
    Expand[y2, x1 - 2] = esq
    Expand[y2, x2] = esq
    Expand = np.fft.fftshift(Expand.copy())
    img_expanded = np.fft.ifft2(Expand.copy())
    if (img.imag == 0).all:
        img_expanded = img_expanded.real

    return img_expanded


def norm(img):
    # norm img values from 0 - 255 in-between 0 - 1
    if (img < 0).any() or (img > 255).any():
        if (img > (-1)).all() and (img < 2).all():
            print('not normed yet')
            img[img < 0] = 0
            img[img > 1] = 1
            return img
        raise NameError('Img can not be normed because its range is out of 0-255.')
    elif (img >= 0).all() and (img <= 1).all():
        # already normed in-between 0 - 1
        return img

    if (img > 1).any():
        if (img < 2).all():
            print('not cut of yet')
            img[img < 0] = 0
            img[img > 1] = 1
            return img
        else:
            img_normed = np.divide(img.copy(), 255)

    return img_normed


class Params:
    """
    Prams class takes parameters as dictionary and allows some other operations.

    """

    def __init__(self, params):
        """

        Args:
            params: parameters as dictionary

        Returns:
            parameters as parameter object

        """
        self.params = {k: np.atleast_2d(v.copy()) for k, v in params.items()}

    def add(self, defect):
        """
        to add defect difference to texture params.

        """
        params_dic = {k: (v.copy() + defect.params[k].copy()) for k, v in self.params.items()}

        params = Params(params_dic)

        return params

    def subtract(self, defect):
        """
        to subtract defect difference from defect params. Return params of texture without defect.

        """
        params_dic = {k: v.copy() - defect.params[k].copy() for k, v in self.params.items()}

        params = Params(params_dic)

        return params

    def average(self, params2):

        params_dic = {k: np.array(v.copy() + params2.params[k].copy()) / 2 for k, v in self.params.items()}

        params = Params(params_dic)

        return params

    def get(self):

        return self.params

    def get_dic(self):

        params_dic = {k: v.copy() for k, v in self.params.items()}

        # for 1-dimensional statistics - cut off last dimension again
        for key in self.params:
            if key == 'magMeans' or key == 'pixelStats' or key == 'varianceHPR':
                params_dic[key] = np.squeeze(self.params[key], axis=0)

                if key == 'varianceHPR':
                    params_dic[key] = self.params[key][0]

        return params_dic
