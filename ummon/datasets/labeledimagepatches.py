# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import warnings
from torchvision import transforms
from imageio import imread
import os
from scipy.ndimage.interpolation import affine_transform

class LabeledImagePatches(Dataset):
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
        * limit (int) : limits the number of patches (default: -1)
        * shuffle (boolean) : random pick limited patches (default: False)
        * oneclass (bool): only return good samples as training examples
        * affine_map (ndarray) : 4x4 ndarray defining the affine transformation
    """
    
    def __init__(self, file, mask_file=None, mode='rgb', mean_normalization=False, train = True, train_percentage=1.0, transform=transforms.Compose([transforms.ToTensor()]),
                 stride_x=16, stride_y=16, window_size=32, train_label=0, test_label=1, crop=[0, 0, -1, -1], limit=-1, shuffle=False, oneclass=False, affine_map=None):
        self.filename = file
        self.img = imread(file)
        self.train = train
        self.train_label = train_label 
        self.test_label = test_label 
        self.train_percentage = train_percentage
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.window_size = window_size
        self.transform = transform
        self.tlx, self.tly, self.brx, self.bry = crop
        self.limit = limit
        self.shuffle = shuffle
        self.oneclass = oneclass
        # Mask image >> only keep the first layer
        self.mask_image = None
        if mask_file is not None:
            self.mask_image = imread(mask_file)[:,:, 0]
            self.mask_image = np.expand_dims(np.asarray(self.mask_image), 2)
        else:
            self.mask_image = np.zeros(self.img.shape[0:2]) 
            self.mask_image = np.expand_dims(self.mask_image, 2)

        assert self.img.shape[:2] == self.mask_image.shape[:2]
        assert self.mask_image.ndim == 3

        # Normalize patch to (H, W, C) with [0, 1] float32
        assert self.img.min() >= 0
        self.img = self.img.astype(np.float32)
        while self.img.max() > 1. : # while as some images are scaled larger than 255 (e.g. disparity images)
            self.img = self.img / 255 
        if self.img.ndim == 3 and self.img.shape[0] < self.img.shape[2] and self.img.shape[0] < self.img.shape[1]:
            # bring channel to back
            self.img = np.transpose(self.img, (1,2,0))
        assert self.img.min() >= 0 and self.img.max() <= 1
        
        # handle gray scale
        if self.img.ndim == 2:
            self.img = np.expand_dims(self.img,2)
        assert self.img.ndim == 3

        # handle transparent channel in png
        if self.img.shape[2] == 4:
            warnings.warn("I dont know how to handle transparency and will skip 4th channel (img[:,:,3]). Shape of image:" + str(self.img.shape), RuntimeWarning)       
            self.img = self.img[:,:,0:3]

        # handle RGB and store a copy
        if mode == 'bgr':
            self.img = self._rgb_to_bgr()
        elif mode == 'gray':
            self.img = self._rgb_to_gray()
        elif mode == 'gray3channel':
            self.img = self._rgb_to_gray3channel()

        # subtract mean per image channel
        if mean_normalization:
            if mode == 'gray':
                self.img[:, :, 0] = self.img[:, :, 0] - np.mean(self.img[:, :, 0])
            else:
                self.img[:, :, 0] = self.img[:, :, 0] - np.mean(self.img[:, :, 0])
                self.img[:, :, 1] = self.img[:, :, 1] - np.mean(self.img[:, :, 1])
                self.img[:, :, 2] = self.img[:, :, 2] - np.mean(self.img[:, :, 2])

            #rescale 01
            self.img = (self.img - np.min(self.img)) / (np.max(self.img) - np.min(self.img))

        # image and mask_image cropping
        if self.brx == -1:
            self.brx = self.img.shape[1]
        if self.bry == -1:
            self.bry = self.img.shape[0]
        self.mask_image = self.mask_image[self.tly: self.bry, self.tlx:self.brx].copy()

        if self.img.ndim == 3:
            self.img      = self.img[self.tly : self.bry, self.tlx:self.brx , :].copy()
        elif self.img.ndim == 2:
            self.img      = self.img[self.tly: self.bry, self.tlx:self.brx].copy()
        else:
            raise AttributeError(self.img.shape + ' image dimensions invalid.')
               
        # Apply given affine transformation to image and mask_image
        if affine_map is not None:
            self.img = affine_transform(self.img, affine_map) 
            self.mask_image = affine_transform(self.mask_image, affine_map)

        # compute some statistics
        self.patches_per_y = (((self.img.shape[0] - self.window_size) // self.stride_y) + 1)
        self.patches_per_x = (((self.img.shape[1] - self.window_size) // self.stride_x) + 1)
        self.num_patches = int(self.patches_per_x) * int(self.patches_per_y)
        
        # compute labels
        self.idx_mapping, self.all_labels, self.num_train_samples, self.num_test_samples = self._label_image()
                  
    def _rgb_to_bgr(self):
        assert self.img.shape[2] == 3
        r = np.expand_dims(self.img[:, :, 0], axis=2)
        g = np.expand_dims(self.img[:, :, 1], axis=2)
        b = np.expand_dims(self.img[:, :, 2], axis=2)
        return np.concatenate((b,g,r), axis=2).copy()
        

    def _rgb_to_gray(self):
        if self.img.shape[2] == 1:
            return self.img
        assert self.img.shape[2] == 3
        r = np.expand_dims(self.img[:, :, 0], axis=2)
        g = np.expand_dims(self.img[:, :, 1], axis=2)
        b = np.expand_dims(self.img[:, :, 2], axis=2)
        return ((.2989 * r) + (.5870 * g) + (.114 * b)).copy()

    def _rgb_to_gray3channel(self):
        img = np.squeeze(self.img)
        if img.ndim == 2:
            g = np.expand_dims(img, axis=2)
            return  np.concatenate((g, g, g), axis=2)   
        if self.img.ndim == 3:
            r = np.expand_dims(self.img[:, :, 0], axis=2)
            g = np.expand_dims(self.img[:, :, 1], axis=2)
            b = np.expand_dims(self.img[:, :, 2], axis=2)
            img = (.2989 * r) + (.5870 * g) + (.114 * b)
            return np.concatenate((img, img, img), axis=2).copy() 
        
        
    def stats(self):
        return {
            "name"  : "ImagePatches",
            "filepath" : self.filename,
            "training split" : self.train_percentage,
            "data set" : "train" if self.train else "test",
            "data samples": len(self),
            "data shape" : self.__getitem__(0)[0].shape,
            "data dtype" : self.__getitem__(0)[0].dtype,
            "data label example" : self.__getitem__(0)[1]
            }
    
    def __repr__(self):
        return str(self.stats())
        
    def __len__(self):
        if self.train:
            return self.num_train_samples
        else:
            return self.num_test_samples  

    def _draw_boarder(self, patch, color, thickness, inplace = True):
         if thickness == 0: return patch
         if inplace == False: patch = patch.copy()
         if patch.shape[2] < 3:
             patch = np.concatenate((patch, patch, patch), axis=2).copy()
         if np.max(patch) > 1 and np.sum(color) <= 3:
             color = np.asarray(color) * 255.
         if np.max(patch) <= 1 and np.sum(color) > 3:
             color =  np.asarray(color) / 255.
         patch[0:thickness,:, :] = color # top
         patch[-thickness:,:, :] = color # bottom
         patch[:,0:thickness, :] = color # left
         patch[:,-thickness:, :] = color # left
         return patch

    def mark_patches(self, patch_indices, color=(255, 255, 255), thickness=2, image = None):
         if isinstance(patch_indices, list) == False and isinstance(patch_indices, np.ndarray) == False: 
            patch_indices = [patch_indices]
         patch_indices = np.asarray(patch_indices).squeeze() if len(patch_indices) > 1 else np.asarray(patch_indices)
         if patch_indices.ndim == 2:
             patch_indices = patch_indices[:,0] #in case of multiple columns take the first
         if image is None:
             # Compute on a copy
             marked_img = self.img.copy()
         else:
             marked_img = image.astype(np.float32)
             if marked_img.max() >= 1:
                 marked_img = marked_img / 255.
             if marked_img.ndim == 2:
                 marked_img = np.expand_dims(marked_img, 2)
             assert self.img.shape[:2] == marked_img.shape[:2]
         for i in patch_indices:
             patch = self._get_patch(i)
             # Draw line..
             patch = self._draw_boarder(patch, color, thickness)
             # Copy to image
             marked_img = self._put_patch(marked_img, i, patch)
         #gray scale
         marked_img = marked_img.squeeze()
         return (marked_img * 255).astype(np.uint8)

    def _label_image(self):
        good_idx, defective_idx = self._compute_labeled_patches(self.limit, self.shuffle)
        idx_mapping = good_idx + defective_idx
        all_labels = np.zeros(self.num_patches).astype(np.int32)
        all_labels[good_idx] = self.train_label
        all_labels[defective_idx] = self.test_label              
        # case one class setting
        if self.oneclass == True:
            if self.train == True:
                idx_mapping = good_idx
            else:
                idx_mapping = defective_idx
                
            # add unlearnt training samples to test set
            if self.train_percentage < 1.0:
                if self.train == True:
                    num_train_samples = int(self.train_percentage * len(good_idx))
                    idx_mapping =  good_idx[:num_train_samples]
                else:
                    num_train_samples = int(self.train_percentage * len(good_idx))
                    unlearnt_training_samples = good_idx[num_train_samples:]
                    idx_mapping =  defective_idx + unlearnt_training_samples

            # store dataset sizes            
            num_train_samples = int(self.train_percentage * len(good_idx))
            num_test_samples = len(defective_idx) + int((1-self.train_percentage) * len(good_idx))
        else:
            # store dataset sizes   
            num_train_samples = int(len(idx_mapping) * self.train_percentage)
            num_test_samples = int(len(idx_mapping) * (1 - self.train_percentage))
        return idx_mapping, all_labels, num_train_samples, num_test_samples

    def _compute_labeled_patches(self, limit=-1, shuffle=False):
        # Handle limits in case we do not want to process the whole image
        if limit == -1:
            limit = self.num_patches
        idx = np.arange(self.num_patches)
        # Randomize the access indices so that we process arbitrary positions
        if shuffle:
            idx = np.random.permutation(idx)
        good_idx, defective_idx = [], []
        # Loop through randomized indices until limit reached and fill label buckets
        for i in range(limit):
            i = idx[i]
            mask = self._get_internal_patch(i, self.mask_image)
            if(np.max(mask) > 0):
                defective_idx.append(i)
            else:
                good_idx.append(i) 
        return good_idx, defective_idx
   

    def _get_internal_patch(self, internal_idx, img):
        x = internal_idx % self.patches_per_x
        y = internal_idx // self.patches_per_x

        topleft_y = y * self.stride_y
        bottomright_y = y * self.stride_y + self.window_size
        topleft_x = x * self.stride_x
        bottomright_x = x * self.stride_x + self.window_size

        patch = img[topleft_y : bottomright_y, topleft_x : bottomright_x, :].copy()
        
        return patch
    
    def _get_patch(self, idx):
        idx = self.idx_mapping[idx]
        return self._get_internal_patch(idx, self.img)
    
    def _put_patch(self, img, idx, patch):
        idx = self.idx_mapping[idx]
        x = idx % self.patches_per_x
        y = idx // self.patches_per_x

        topleft_y = y * self.stride_y
        bottomright_y = y * self.stride_y + self.window_size
        topleft_x = x * self.stride_x
        bottomright_x = x * self.stride_x + self.window_size
        
        if patch.ndim == 2: np.expand_dims(patch, 2)
        if patch.shape[2] > img.shape[2]:
             img = np.concatenate((img, img, img), axis=2).copy()
        img[topleft_y : bottomright_y, topleft_x : bottomright_x, :] = patch
        return img
    
    def _get_label(self, idx):
         idx = self.idx_mapping[idx]
         return int(self.all_labels[idx])

    def __getitem__(self, idx):
        patch = self._get_patch(idx)
        label = self._get_label(idx)
        if self.transform:
            patch = self.transform(patch)
        return patch, label
