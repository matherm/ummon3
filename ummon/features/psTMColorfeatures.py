import sys

import PIL.Image
import numpy as np
import torch
from ummon.features.ps_color.textureColorAnalysis import TextureColorAnalysis

__all__ = ['PSTMColorfeatures']


class PSTMColorfeatures:
    """
    Extracts Portilla & Simoncelli Texture Model features.

    [1] Portilla & Simoncelli (2000). A parametric texture model based on joint statistics of complex wavelet coefficients.
        http://www.cns.nyu.edu/pub/lcv/portilla99-reprint.pdf

    Usage
    ======
        transform = PSTMColorfeatures()
        transform(gsImage)

        OR

        psfeatures = PSTMColorfeatures()
        my_transforms = transforms.Compose([transforms.ToPILImage(), PSTMColorfeatures])
        test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                            train=False, \
                            transform=my_transforms)


    """
    def __init__(self, scales=4, number_of_orientations=4, spatial_neighborhood=7):
        self._nsc = scales
        self._nor = number_of_orientations
        self._na = spatial_neighborhood

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)

        assert img.ndim == 3 and img.shape[2] == 3

        psFeatures = TextureColorAnalysis(img, self._nsc, self._nor, self._na)
        psFeatures.analyze()
        return torch.from_numpy(psFeatures.getFeaturesArray().astype(np.float32))
