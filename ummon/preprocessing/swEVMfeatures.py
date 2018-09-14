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
    Extracts Schutt Wichmann Early Vision Model (EVM) features (Magnitude of the complex model output). Currently a fixed V1 parameter set is used.

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

    def __init__(self, pooling_mode='None', normalized=True, gram=False, gram_diagonal=False):
        """
        Parameters
        ----------
            *normalized (bool) : compute normalized features
            *meanFreqOutput (bool)  : compute mean over frequency

        """
        self.normalized = normalized
        self.gram = gram
        self.gram_diagonal = gram_diagonal
        self.pooling_mode = pooling_mode

    def __call__(self, image):

        if isinstance(image, torch.Tensor):
            image = image.data.numpy()

        image = np.squeeze(image)
        assert image.ndim == 2

        evm = swEMV()

        if self.normalized:
            out = evm.V1(image)

            if self.pooling_mode == 'mean_freq_orient':
                out = np.mean(np.mean(out, 0), 0)
            elif self.pooling_mode == 'mean_im_space':
                out = np.mean(out, axis=(2,3))
            elif self.pooling_mode == 'max_freq_orient':
                out = np.max(np.max(out, 0), 0)
            elif self.pooling_mode == 'max_im_space':
                out = np.max(out, axis=(2, 3))
            elif self.pooling_mode == 'max_combined':
                out = np.concatenate((np.max(out, axis=(2, 3)).flatten(), np.max(np.max(out, 0), 0).flatten()))
            elif self.pooling_mode == 'mean_combined':
                out = np.concatenate((np.mean(out, axis=(2, 3)).flatten(), np.mean(np.mean(out, 0), 0).flatten()))

        else:
            # retrun magnitude
            out, _, _ = evm.decomp_Gabor(image)
            out = np.abs(out)

            if self.pooling_mode == 'mean_freq_orient':
                out = np.mean(np.abs(out, 0), 0)
            elif self.pooling_mode == 'mean_im_space':
                out = np.mean(out, axis=(2,3))
            elif self.pooling_mode == 'max_freq_orient':
                out = np.max(np.abs(out, 0), 0)
            elif self.pooling_mode == 'max_im_space':
                out = np.max(out, axis=(2, 3))

        if self.gram:
            v = np.expand_dims(out.flatten(), axis=1)
            out = v * v.T
        elif self.gram_diagonal:
            v = np.expand_dims(out.flatten(), axis=1)
            out = v ** 2

        # magnitude of the complex model output.
        return torch.from_numpy(np.asarray(out.flatten(), dtype='float32'))
