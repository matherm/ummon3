import sys
sys.path.insert(0, '../prog')

import numpy as np
import scipy.io

from .textureColorAnalysis import TextureColorAnalysis

def test_PSColor():
    from imageio import imread
    img = imread('~/matlab/portilla-simoncelli/colorTextureSynth/olives256.o.bmp')[115:243, 72:200, :]

    analysis = TextureColorAnalysis(img, 3)
    analysis.analyze()

    pixelLPStatsML = scipy.io.loadmat('/home/fabian/matlab/pixelLPStats.mat')['statsLPim']
    autoCorrRealML = scipy.io.loadmat('/home/fabian/matlab/autoCorrReal.mat')['acr']
    autoCorrMagML = scipy.io.loadmat('/home/fabian/matlab/autoCorrMag.mat')['ace']
    magMeansML = scipy.io.loadmat('/home/fabian/matlab/magMeans.mat')['magMeans0']
    cousinMagCorrML = scipy.io.loadmat('/home/fabian/matlab/cousinMagCorr.mat')['C0']
    parentMagCorrML = scipy.io.loadmat('/home/fabian/matlab/parentMagCorr.mat')['Cx0']
    cousinRealCorrML = scipy.io.loadmat('/home/fabian/matlab/cousinRealCorr.mat')['Cr0']
    parentRealCorrML = scipy.io.loadmat('/home/fabian/matlab/parentRealCorr.mat')['Crx0']
    varianceHPRML = scipy.io.loadmat('/home/fabian/matlab/varianceHPR.mat')['vHPR0']
    colorCorrML = scipy.io.loadmat('/home/fabian/matlab/colorCorr.mat')['Cclr0']

    np.testing.assert_allclose(analysis.pixelLPStats, pixelLPStatsML, atol=1e-10)
    np.testing.assert_allclose(analysis.autoCorrReal, autoCorrRealML, atol=1e-10)
    np.testing.assert_allclose(analysis.autoCorrMag, autoCorrMagML, atol=1e-10)
    np.testing.assert_allclose(analysis.magMeans, magMeansML, atol=1e-10)
    np.testing.assert_allclose(analysis.cousinMagCorr, cousinMagCorrML, atol=1e-10)
    np.testing.assert_allclose(analysis.parentMagCorr, parentMagCorrML, atol=1e-10)
    np.testing.assert_allclose(analysis.cousinRealCorr, cousinRealCorrML, atol=1e-10)
    np.testing.assert_allclose(analysis.parentRealCorr, parentRealCorrML, atol=1e-10)
    np.testing.assert_allclose(analysis.varianceHPR, varianceHPRML, atol=1e-10)
    np.testing.assert_allclose(analysis.colorCorr, colorCorrML, atol=1e-10)

if __name__ == '__main__':
    test_PSColor()