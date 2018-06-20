from __future__ import division

import numpy as np

from .auxiliary import loadFromMat, compare


def compare_coefficients(coeff):
    ## compare coefficients with matlab
    coeff_pyr = loadFromMat('pyr0', 0)
    coeff_ind = loadFromMat('pind0', 0)
    #coeff_mat = loadFromMat('pyr0', False)

    coeff_mat = []

    # reconstruct coefficients from pyr vector and ind of matlab
    ind = 0
    for shapes in coeff_ind:
        shapes = shapes.astype(np.uint32)
        tmp = np.reshape(coeff_pyr[ind:ind+shapes[0]*shapes[1]], shapes, order='F')
        coeff_mat.append(tmp)
        ind = ind + shapes[0]*shapes[1] # - 1

    # reconstruct subbands of matlab - to get same representation of coefficients as in python pyramid
    coeff_list = []
    band_list = []
    #band_i = 0
    first = 1
    nband = 0
    for band in coeff_mat:
        if first:
            coeff_list.append([band])
            first = 0
        else:
            if (nband)%4==0 and nband!=0:
                coeff_list.append(band_list)
                band_list = []
                band_list.append(band)
            else:
                band_list.append(band)
            nband = nband+1

    # compare coefficients: matlab - python pyramid
    coeff_list.append([band])

    compare(coeff[0][0], coeff_list[0][0], "coeff[0][0]")

    compare(coeff[1][0], coeff_list[1][0], "coeff[1][0]")
    compare(coeff[1][1], coeff_list[1][1], "coeff[1][1]")
    compare(coeff[1][2], coeff_list[1][2], "coeff[1][2]")
    compare(coeff[1][3], coeff_list[1][3], "coeff[1][3]")

    compare(coeff[2][0], coeff_list[2][0], "coeff[2][0]")
    compare(coeff[2][1], coeff_list[2][1], "coeff[2][1]")
    compare(coeff[2][2], coeff_list[2][2], "coeff[2][2]")
    compare(coeff[2][3], coeff_list[2][3], "coeff[2][3]")

    compare(coeff[3][0], coeff_list[3][0], "coeff[3][0]")
    compare(coeff[3][1], coeff_list[3][1], "coeff[3][1]")
    compare(coeff[3][2], coeff_list[3][2], "coeff[3][2]")
    compare(coeff[3][3], coeff_list[3][3], "coeff[3][3]")

    compare(coeff[4][0], coeff_list[4][0], "coeff[4][0]")
    compare(coeff[4][1], coeff_list[4][1], "coeff[4][1]")
    compare(coeff[4][2], coeff_list[4][2], "coeff[4][2]")
    compare(coeff[4][3], coeff_list[4][3], "coeff[4][3]")

    compare(coeff[5][0], coeff_list[5][0], "coeff[5][0]")


def compare_reconstruction(img):


    fake_img_mat = loadFromMat("ch")
    compare(img, fake_img_mat, "fake image")


