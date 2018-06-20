from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

from .auxiliary import compare, scale2

def plotOrients(corr, key, nsc, binar=False):
    """
    Helps to plot multiple orientation statistics in a notebook later.

    """
    print('nsc: ', nsc)
    u = 151
    #for nsc in range(params['autoCorrMag'].shape[2]):
    for nor in range(corr.shape[3]):
        if binar:
            plt.subplot(u),plt.imshow(corr[:,:,nsc,nor],  vmin=0, vmax=1, cmap = 'gray')
        else:
            plt.subplot(u),plt.imshow(corr[:,:,nsc,nor], cmap = 'gray')
        plt.title(key+'_'+str(nor)), plt.xticks([]), plt.yticks([])
        u = u+1
    fig = plt.gcf()
    fig.set_size_inches(10,20)


def plotScales(corr, key, binar=False):
    u = 151
    #for nsc in range(params['autoCorrMag'].shape[2]):
    for nsc in range(corr.shape[2]):
        if binar:
            plt.subplot(u),plt.imshow(corr[:,:,nsc], vmin=0, vmax=1, cmap = 'gray')
        else:
            plt.subplot(u),plt.imshow(corr[:,:,nsc], cmap = 'gray')
        plt.title(key+'_'+str(nsc)), plt.xticks([]), plt.yticks([])
        u = u+1
    fig = plt.gcf()
    fig.set_size_inches(10,20)


def f_compare(x,y, reltol=0, abstol=1.e-1):
    dic = dict(measure='percentual_close',rtol=reltol, atol=abstol)
    return {k:compare(x[k],y[k],k, **dic) for k in x}

def f_diff(x,y, reltol=0, abstol=1.e-1):
    dic = dict(measure='square_diff',rtol=reltol, atol=abstol)
    return {k:np.sum(np.absolute(compare(x[k],y[k],k, **dic))) for k in x}

def f_normed_diff(x,y, reltol=0, abstol=1.e-1):
    distance = {}
    for key in x:
        distance[key] = np.sum(compare(x[key], y[key], key, normed=True, measure='square_diff',rtol=reltol, atol=abstol))
    return distance