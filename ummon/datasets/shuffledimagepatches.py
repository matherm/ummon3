# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings
from torchvision import transforms
from imageio import imread
from ummon.datasets.imagepatches import *
from ummon.datasets.labeledimagepatches import *
import numpy as np    
    
class ShuffledImagePatches(ImagePatches):
    
    """
    Dataset for generating data from a single given image. It used a window-scheme, hence the name ImageTiles.
    
    Arguments:
        * file (str) : The image filename
        * mode (str) : The processing mode 'bgr' or 'gray' or 'rgb' or 'gray3channel' (default="bgr")
        * train (bool) : train or test set
        * train_percentage (float) : percentage of train patches compared to all patches
        * transform (torchvision.transforms) : Image Transformations (default transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        * stride_x : The overlap in x direction
        * stride_y : The overlap in y direction
        * window_size (int) : square size of the resulting patches
        * crop (list) : tlx, tly, brx, bry, default [0, 0, -1, -1]
        * mode (str): "full", "per_channel"
        * limit (int) : patch limit (default: -1)
    """
    
    def __init__(self, *args, shuffle_mode="full", **kwargs):
        super().__init__(*args, **kwargs)
        
        # shuffling the underlying image data responsible for producing patches
        self.shuffle_(self.img, shuffle_mode)
        
    def shuffle_(self, image, mode="full"):
        """
        Arguments:
            * image (ndarray)
            * mode (str): "full", "per_channel"
        """
        assert mode in ["full", "per_channel"]
        
        if mode == "per_channel" and image.ndim == 3:
            np.random.shuffle(image.reshape(-1, image.shape[2]))
        else:
            np.random.shuffle(image.reshape(-1))
    
        
class ShuffledOCLabeledImagePatches(OCLabeledImagePatches):
    """
    Dataset for generating data from a single given image with labeled defects in a separate mask file. 
    It used a window-scheme, hence the name Image Patches.
    
    Arguments:
        * file (str) : The image filename
        * file (str) : The mask image filename
        * mode (str) : The processing mode 'bgr' or 'gray' or 'rgb' or 'gray3channel' (default="bgr")
        * train (bool) : train or test set
        * train_percentage (float) : percentage of train patches compared to all patches
        * transform (torchvision.transforms) : Image Transformations (default transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        * stride_x : The overlap in x direction
        * stride_y : The overlap in y direction
        * window_size (int) : square size of the resulting patches
        * crop (list) : tlx, tly, brx, bry, default [0, 0, -1, -1]
    """
    def __init__(self, *args, shuffle_mode="full", **kwargs):
        super().__init__(*args, **kwargs)
    
        self.shuffled_patches_(self.img, shuffle_mode)
        
    def shuffled_patches_(self, image, mode="full"): 
        """
        Arguments:
            * image (ndarray)
            * mode (str): "full", "per_channel"
        """
        assert mode in ["full", "per_channel"]
        
        # Stich the image patches to a single large image
        stiched_image = np.zeros( (len(self),) + self._get_patch(0).shape )
        for i in range(len(self)):
            stiched_image[i] = self._get_patch(i)
            
        # shuffle magic
        if mode == "per_channel" and stiched_image.ndim == 4:
            np.random.shuffle(stiched_image.reshape(-1, stiched_image.shape[3]))
        else:
            np.random.shuffle(stiched_image.reshape(-1))
            
        # replace the original patches with the shuffled ones
        for i in range(len(stiched_image)):
            self._put_patch(image, i, stiched_image[i])
                
    
    
    
       
        
        