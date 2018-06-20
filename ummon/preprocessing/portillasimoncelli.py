import torch
import numpy as np
from .portilla_simoncelli.analysis import Analysis

__all__ = [ 'PortillaSimoncelli']


class PortillaSimoncelli():
    
    def __init__(self, scales = 2, list_stats=["pixelStats", "pixelLPStats"]):
        self.scales = scales
        self.list_stats = list_stats

    def __call__(self, image):
        '''
        This method computes the portilla simoncelli texture features.
        [1] Portilla & Simoncelli (2000). A parametric texture model based on joint statistics of complex wavelet coefficients.
            http://www.cns.nyu.edu/pub/lcv/portilla99-reprint.pdf
        
        Note
        ------
         Possible statistics are {"pixelStats": 
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
        Image as [Channels, Height, Width]
        
        
        Returns
        -------
        PS Feature statistics
        '''
        assert isinstance(image, torch.Tensor)
        assert image.dim() == 2
        
        # Analysis
        a = Analysis(image.numpy(), self.scales) # img,nsc # number of bandpasses (scales)
        a.computeFeatures()
        params = a.getFeatures().get_dic()
        return torch.cat([torch.from_numpy(params[stat]).float().view(-1) for stat in self.list_stats])

