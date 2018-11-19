from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings
from torchvision import transforms
from ummon.preprocessing.anomaly import *
from ummon.datasets.imagepatches import ImagePatches
import numpy as np

class AnomalyImagePatches(ImagePatches):
     """
     
     Dataset for generating data and anomaly from a single given image. It used a window-scheme, hence the name ImageTiles.
     
     Note
     ====
     Compared to ImagePatches all anomaly patches have label -1 (or as specified with anomaly_label) whereas non-anomaly patches have 1
    
     Arguments:
        * file (str) : The image filename
        * mode (str) : The processing mode 'bgr' or 'gray' or 'rgb' or 'gray3channel' (default="bgr")
         * train (bool) : train or test set
        * train_percentage (float) : percentage of train patches compared to all patches
        * transform (torchvision.transforms) : Image Transformations (default transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        * stride_x : The overlap in x direction
        * stride_y : The overlap in y direction
        * window_size (int) : square size of the resulting patches
        * anomaly (torchvision.transformation)
        * permutation (string) : permutation mode 'random' = random position, 'center' = fixed center, 'full' = all posible positions..
        * propability (float) : propability of a anomaly
        * anomaly_label (int) : the label for anomaly data (normal data is 0)
        * crop (list) : tlx, tly, brx, bry, default [0, 0, -1, -1]
     """
     def __init__(self, file, mode='bgr', mean_normalization=False, train = True, train_percentage=.8, transform=transforms.Compose([transforms.ToTensor()]),
                 stride_x=16, stride_y=16, window_size=32, anomaly=SquareAnomaly(), permutation='random', propability=0.2, label=1, anomaly_label = -1, crop=[0, 0, -1, -1]):
        super(AnomalyImagePatches, self).__init__(file, mode, mean_normalization, train, train_percentage, transform, stride_x, stride_y, window_size, label, crop)

        assert anomaly_label == 0 or anomaly_label == -1

        self.permutation = permutation
        self.propability = propability
        self.anomaly_label = anomaly_label
        self.anomaly = anomaly
        if self.permutation == 'full':
            if isinstance(self.anomaly, LineDefectAnomaly):
                self.anomaly_permutations = (self.window_size - self.anomaly.anom_size + 1)
            else:
                raise NotImplementedError(str(self.anomaly) + ' not implemented yet.')
        elif self.permutation == 'center':
            if isinstance(self.anomaly, LineDefectAnomaly):
                self.anomaly_permutations = 1
            else:
                raise NotImplementedError(str(self.anomaly) + ' not implemented yet.')
        elif self.permutation == 'random':
            self.anomaly_permutations = 1
        else:
            raise NotImplementedError(str(self.permutation) + ' not implemented yet.')

    
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
             patch, _, anom_idx = self.get_anomaly_patch(i)
             # Draw boarder..
             patch = self._draw_boarder(patch, color, thickness)
             # Copy to image
             marked_img = self._put_patch(marked_img, anom_idx, patch)
         #gray scale
         marked_img = marked_img.squeeze()
         return (marked_img * 255).astype(np.uint8)
             
     def _compute_anomaly(self, patch, pos):
         if isinstance(self.anomaly, LineDefectAnomaly):
            return self.anomaly(patch, pos)
         else:
             return self.anomaly(patch)

     def _get_anomaly_position(self, idx):
        if self.permutation == 'full':
            pos = idx % self.anomaly_permutations
        elif self.permutation == 'center':
            pos = self.window_size // 2 # compute horizontal center
        else:
            pos = -1 # random position of anomaly
        return pos

     def get_anomaly_patch(self, idx):
        patch = self._get_patch(idx // self.anomaly_permutations)
        pos = self._get_anomaly_position(idx)
        
        # Add anomaly with propability given by self.propability and label -1 respectivly
        if np.random.rand() < self.propability:
            patch = self._compute_anomaly(patch, pos)
            label = self.anomaly_label
        else:
            label = self.label
        anom_idx = idx // self.anomaly_permutations
        return patch, label, anom_idx

     def __getitem__(self, idx):
        patch, label, anom_idx = self.get_anomaly_patch(idx)
        if self.transform:
            return self.transform(patch), label
        return patch, label

     def __len__(self):
        return super(AnomalyImagePatches, self).__len__() * self.anomaly_permutations

     def __repr__(self):
         return super(AnomalyImagePatches, self).__repr__()

