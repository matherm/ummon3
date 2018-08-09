#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Michael Grunwald at 09.08.2018
"""

import numpy as np
import torch

from .schuett_wichmann_evm.swEVM import swEMV

__all__ = ['swEVMfeatures']


class swEVMfeatures():
    """
    Extracts SW EVM features (Magnitude of the complex model output). Currently a fixed V1 parameter set is used.

    Schuett, H. H., & Wichmann, F. A. (2017). An image-computable psychophysical spatial vision model. Journal of Vision, 17(12):12, 1–35, doi:10.1167/17.12.12.

    Usage
    ======
            transform = swEVMfeatures(normalized=True, meanFreq)
            transform(luminanceImage)

            OR

            evm = swEVMfeatures(normalized=True)
            my_transforms = transforms.Compose([transforms.ToTensor(), evm])
            test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms)


    Input
    ======
        *Luminance ímage as [Height, Width]

    Return
    =====
        *feature (torch.Tensor) Magnitude of the complex model output.

    """

    def __init__(self, normalized=True, meanFreqOutput=False):
        """
        Parameters
        ----------
            *normalized (bool) : compute normalized features
            *meanFreqOutput (bool)  : compute mean over frequency

        """
        self.normalized = normalized
        self.meanFreqOutput = meanFreqOutput

    def __call__(self, image):

        if isinstance(image, torch.Tensor):
            image = image.data.numpy()

        image = np.squeeze(image)
        assert image.ndim == 2

        evm = swEMV()

        if self.normalized:
            out, _ = evm.V1(image)
        else:
            out, _, _ = evm.decomp_Gabor(image)

        if self.meanFreqOutput:
            out = np.flip(np.mean(np.abs(np.mean(out, 0)), 0), 1)

        # magnitude of the complex model output.
        return torch.from_numpy(np.abs(out).astype('float32').flatten())
