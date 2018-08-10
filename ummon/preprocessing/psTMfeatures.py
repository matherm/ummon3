#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created by Michael Grunwald at 09.08.2018
"""

import numpy as np
import torch
from .portilla_simoncelli_tm.analysis import Analysis

__all__ = ['PSTMfeatures']


class PSTMfeatures():
    """
        Extracts Portilla & Simoncelli Texture Model features.

        [1] Portilla & Simoncelli (2000). A parametric texture model based on joint statistics of complex wavelet coefficients.
            http://www.cns.nyu.edu/pub/lcv/portilla99-reprint.pdf

        Usage
        ======
            transform = PSTMfeatures()
            transform(gsImage)

            OR

            psfeatures = PSTMfeatures()
            my_transforms = transforms.Compose([transforms.ToTensor(), psfeatures])
            test_set = ImagePatches("ummon/datasets/testdata/Wood-0035.png", \
                                train=False, \
                                transform=my_transforms)


        """
    
    def __init__(self, scales = 2):

        self.scales = scales

    def __call__(self, image):
        '''
        This method computes the portilla simoncelli texture features.

        
        Note
        ------
         Joint statistics   {"pixelStats":
                             "pixelLPStats":
                             "autoCorrReal":
                             "autoCorrMag":
                             "magMeans":
                             "cousinMagCorr"
                             "parentMagCorr"
                             "cousinRealCorr"
                             "parentRealCorr"
                             "varianceHPR"}
        
        Arguments
        --------
        Gray scale image as [Height, Width]
        
        
        Returns
        -------
        PS Feature statistics
        '''
        if isinstance(image, torch.Tensor):
            image = image.numpy()

        image = np.squeeze(image)
        assert image.ndim == 2

        # Todo: check scales and image size.
        
        # Analysis
        a = Analysis(image, nsc=self.scales) # img,nsc # number of bandpasses (scales)
        a.computeFeatures()
        jointStat = a.getJointStatisticsFeatures()
        return  torch.from_numpy(jointStat.astype('float32'))

