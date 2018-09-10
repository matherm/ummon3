from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings

__all__ = ["UnsupTensorDataset" , "SiameseTensorDataset" , "WeightedPair", "Triplet", "ImagePatches", "AnomalyImagePatches", "TripletTensorDataset", "NumpyDataset"]

class UnsupTensorDataset(Dataset):
    """Dataset wrapping tensors for unsupervised trainings.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
    """

    def __init__(self, data_tensor):
        self.data_tensor = data_tensor

    def __getitem__(self, index):
        return self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)
    
    
class SiameseTensorDataset(Dataset):
    """Dataset wrapping tensors for siamese networks.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor_l (Tensor): contains sample data.
        data_tensor_r (Tensor): contains sample data.
        target_tensor_l (Tensor): contains target data left.
        target_tensor_r (Tensor): contains target data right.
    """

    def __init__(self, data_tensor_l, data_tensor_r, target_tensor_l, target_tensor_r):
        self.data_tensor_l = data_tensor_l
        self.data_tensor_r= data_tensor_r
        self.target_tensor_l = target_tensor_l
        self.target_tensor_r = target_tensor_r
        assert self.data_tensor_l.size(0) == self.data_tensor_r.size(0) == self.target_tensor_l.size(0) == self.target_tensor_r.size(0)

    def __getitem__(self, index):
        return (self.data_tensor_l[index], self.data_tensor_r[index]), (self.target_tensor_l[index], self.target_tensor_r[index])

    def __len__(self):
        return self.data_tensor_l.size(0)
    
    
import torch
from torch.utils.data import Dataset
import numpy as np
class WeightedPair(Dataset):
    """
    Dataset for training with deep metric networks (Siamese).
    
    It takes a dataset like cifar10 and ensures that balanced tuples are returnd.
    Balanced means that there are same number of same lables and different labels.
    
    Arguments:
        *dataset (torch.dataset) : The underlying dataset
        *class_ration :     The ratio of matching and unmatching labels
    
    """
    def __init__(self, dataset, class_ratio=0.5):
        self.dataset = dataset
        self.class_ratio = float(class_ratio)
        self.class_idx = {}
        if self.dataset.train:
            self.labels = self.dataset.train_labels
        else:
            self.labels = self.dataset.test_labels
        for idx, c in enumerate(self.labels):
            if c in self.class_idx.keys():
                self.class_idx[c].append(idx)
            else:
                self.class_idx[c] = [idx]

    def __getitem__(self, index):
        left_img, left_label = self.dataset[index]

        if str(self.class_ratio) == "1.0" or str(self.class_ratio) != "0.0" and torch.rand(1)[0] <= self.class_ratio:
            # same label
            count_label = len(self.class_idx[left_label])
            rnd_idx_same_label = np.random.randint(count_label, size=1)[0]
            right_img, right_label = self.dataset[self.class_idx[left_label][rnd_idx_same_label]]
        else:
            # different label
            rnd_idx = np.random.randint(len(self.labels), size=1)[0]
            # roll until not same
            while rnd_idx in self.class_idx[left_label]:
                rnd_idx = np.random.randint(len(self.labels), size=1)[0]

            right_img, right_label = self.dataset[rnd_idx]

        return (left_img, right_img), (left_label, right_label)

    def __len__(self):
        return len(self.dataset)


class Triplet(Dataset):
    """
    Dataset for training with deep metric networks (Triplet).
    
    It takes a dataset like cifar10 and ensures that balanced tuples are returnd.
    Balanced means that there are same number of same lables and different labels.
    
    Arguments:
        *dataset (torch.dataset) : The underlying dataset
        *class_ration :     The ratio of matching and unmatching labels
    
    """
    def __init__(self, dataset):
        self.dataset = dataset
        self.class_idx = {}
        if self.dataset.train:
            self.labels = self.dataset.train_labels
        else:
            self.labels = self.dataset.test_labels
        for idx, c in enumerate(self.labels):
            if c in self.class_idx.keys():
                self.class_idx[c].append(idx)
            else:
                self.class_idx[c] = [idx]

    def __getitem__(self, index):
        # ANCHOR
        anchor, anchor_label = self.dataset[index]

        # POSITIV        
        positive_count = len(self.class_idx[anchor_label])
        poitive_rand   = np.random.randint(0, positive_count)
        positive = self.dataset[self.class_idx[anchor_label][poitive_rand]]
        pos, pos_label = positive[0], positive[1]

        # NEGATIVE
        negative_count = len(self.labels) - positive_count
        negative_rand  = np.random.randint(0, negative_count)
        negative = self.dataset[negative_rand]
        while negative[1] == anchor_label:
            negative_rand  = np.random.randint(0, negative_count)
            negative = self.dataset[negative_rand]
        neg, neg_label = negative[0], negative[1]
        
        return (anchor, pos, neg), (anchor_label, pos_label, neg_label)

    def __len__(self):
        return len(self.dataset)


from torchvision import transforms
from imageio import imread
class ImagePatches(Dataset):
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
    """
    
    def __init__(self, file, mode='bgr', train = True, train_percentage=.8, transform=transforms.Compose([transforms.ToTensor()]),
                 stride_x=16, stride_y=16, window_size=32, label=1, crop=[0, 0, -1, -1]):

        self.filename = file
        self.img = imread(file)
        self.train = train
        self.train_percentage = train_percentage
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.window_size = window_size
        self.transform = transform
        self.label = label

        self.tlx, self.tly, self.brx, self.bry = crop

        # Normalize patch to (H, W, C) with [0, 1] float32
        assert self.img.min() >= 0
        self.img = self.img.astype(np.float32)
        if self.img.max() > 1. :
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
        self.img_orig = self.img
        if mode == 'bgr':
            self.img = self.__rgb_to_bgr__()
        elif mode == 'gray':
            self.img = self.__rgb_to_gray__()
        elif mode == 'gray3channel':
            self.img = self.__rgb_to_gray3channel__()

        if self.brx == -1:
            self.brx = self.img.shape[1]
        if self.bry == -1:
            self.bry = self.img.shape[0]

        if self.img.ndim == 3:
            self.img      = self.img[self.tly : self.bry, self.tlx:self.brx , :].copy()
            self.img_orig = self.img_orig[self.tly : self.bry, self.tlx:self.brx , :].copy()
        elif self.img.ndim == 2:
            self.img      = self.img[self.tly: self.bry, self.tlx:self.brx].copy()
            self.img_orig = self.img_orig[self.tly: self.bry, self.tlx:self.brx].copy()
        else:
            raise AttributeError(self.img.shape + ' image dimensions invalid.')

        self.patches_per_y = (((self.img.shape[0] - self.window_size) // self.stride_y) + 1)
        self.patches_per_x = (((self.img.shape[1] - self.window_size) // self.stride_x) + 1)
        self.dataset_size = int(self.patches_per_y) * int(self.patches_per_x)

    def __rgb_to_bgr__(self):
        assert self.img.shape[2] == 3
        r = np.expand_dims(self.img[:, :, 0], axis=2)
        g = np.expand_dims(self.img[:, :, 1], axis=2)
        b = np.expand_dims(self.img[:, :, 2], axis=2)
        return np.concatenate((b,g,r), axis=2).copy()
        

    def __rgb_to_gray__(self):
        if self.img.shape[2] == 1:
            return self.img
        assert self.img.shape[2] == 3
        r = np.expand_dims(self.img[:, :, 0], axis=2)
        g = np.expand_dims(self.img[:, :, 1], axis=2)
        b = np.expand_dims(self.img[:, :, 2], axis=2)
        return ((.2989 * r) + (.5870 * g) + (.114 * b)).copy()

    def __rgb_to_gray3channel__(self):
        if self.img.ndim == 2:
            g = np.expand_dims(self.img, axis=2)
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
            "data split" : self.train_percentage,
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
            return int(self.train_percentage * self.dataset_size)
        else:
            return int((1 - self.train_percentage) * self.dataset_size)

    def _draw_boarder(self, patch, color, thickness, inplace = True):
         if inplace == False: patch = patch.copy()
         if np.max(patch) > 1 and color <= 1:
             color = color * 255.
         if np.max(patch) <= 1 and color > 1:
             color = color / 255.
         patch[0:thickness,:, :] = color # top
         patch[-thickness:,:, :] = color # bottom
         patch[:,0:thickness, :] = color # left
         patch[:,-thickness:, :] = color # left
         return patch

    def mark_patches(self, patch_indices, color=255, thickness=2, mode="rgb"):
         if isinstance(patch_indices, list) == False: patch_indices = [patch_indices]
         # compute a copy of our image as a backup
         original_img = self.img
         self.img = original_img.copy()
         for i in patch_indices:
             patch, _ = self._get_patch(i)
             # Draw white line..
             # ..as we operate inplace this will change the underlying self.img
             self._draw_boarder(patch, color, thickness, inplace = True)
             # Copy to image
             self._put_patch(i, patch)
         # restore original image
         marked_img = self.img
         self.img = original_img
         if marked_img.shape[2] == 1: #gray scale
             marked_img = marked_img.squeeze(2)
         return (marked_img * 255).astype(np.uint8)


    def _get_patch(self, idx):
        x = idx % self.patches_per_x
        y = idx // self.patches_per_x

        topleft_y = y * self.stride_y
        bottomright_y = y * self.stride_y + self.window_size
        topleft_x = x * self.stride_x
        bottomright_x = x * self.stride_x + self.window_size

        patch = self.img[topleft_y : bottomright_y, topleft_x : bottomright_x, :].copy()
        
        return patch
    
    def _put_patch(self, idx, patch):
        x = idx % self.patches_per_x
        y = idx // self.patches_per_x

        topleft_y = y * self.stride_y
        bottomright_y = y * self.stride_y + self.window_size
        topleft_x = x * self.stride_x
        bottomright_x = x * self.stride_x + self.window_size

        self.img[topleft_y : bottomright_y, topleft_x : bottomright_x, :] = patch
        
        return patch

    def __getitem__(self, idx):
        patch = self._get_patch(idx)
        label = self.label
        if self.transform:
            patch = self.transform(patch)
        return patch, label
    
from ummon.preprocessing.anomaly import *
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
     def __init__(self, file, mode='bgr', train = True, train_percentage=.8, transform=transforms.Compose([transforms.ToTensor()]),
                 stride_x=16, stride_y=16, window_size=32, anomaly=SquareAnomaly(), permutation='random', propability=0.2, label=1, anomaly_label = -1, crop=[0, 0, -1, -1]):
        super(AnomalyImagePatches, self).__init__(file, mode, train, train_percentage, transform, stride_x, stride_y, window_size, label, crop)

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

    
     def mark_patches(self, patch_indices, color=255, thickness=2):
         if isinstance(patch_indices, list) == False and isinstance(patch_indices, np.ndarray) == False: 
             patch_indices = [patch_indices]
           # compute a copy of our image as a backup
         original_img = self.img
         self.img = original_img.copy()
         for i in patch_indices:
             patch, _ = self.get_anomaly_patch(i)
             # Draw white line..
             # ..as we operate inplace this will change the underlying self.img
             self._draw_boarder(patch, color, thickness, inplace = True)
             # Copy to image
             self._put_patch(i // self.anomaly_permutations, patch)
         # restore original image
         marked_img = self.img
         self.img = original_img
         if marked_img.shape[2] == 1: #gray scale
             marked_img = marked_img.squeeze(2)
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

        return patch, label

     def __getitem__(self, idx):
        patch, label = self.get_anomaly_patch(idx)
        if self.transform:
            return self.transform(patch), label
        return patch, label

     def __len__(self):
        return super(AnomalyImagePatches, self).__len__() * self.anomaly_permutations

     def __repr__(self):
         return super(AnomalyImagePatches, self).__repr__()

    
class TripletTensorDataset(Dataset):
    """Dataset wrapping tensors for triplet networks.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor_l (Tensor): contains sample data.
        data_tensor_m (Tensor): contains sample data.
        data_tensor_r (Tensor): contains sample data.
        target_tensor (Tensor): contains target data.
    """

    def __init__(self, data_tensor_l, data_tensor_m, data_tensor_r, target_tensor):
        self.data_tensor_l = data_tensor_l
        self.data_tensor_m = data_tensor_m
        self.data_tensor_r= data_tensor_r
        self.target_tensor = target_tensor

    def __getitem__(self, index):
        return (self.data_tensor_l[index], self.data_tensor_m[index], self.data_tensor_r[index]), self.target_tensor[index]

    def __len__(self):
        assert self.data_tensor_l.size(0) == self.data_tensor_m.size(0) == self.data_tensor_r.size(0) == self.target_tensor.size(0)
        return self.data_tensor_l.size(0)
    
    
class NumpyDataset(Dataset):
    """
    Small dataset used to wrap a existing dataset.
    
    Usage
        *NumpyDataset(MNIST(..)).labels => [Labels]
        *NumpyDataset(MNIST(..)).data => [Samples, Features]
        
    Parameters
    *dataset (torch.utils.data.Dataset) : The torch dataset to convert
    *only_labels (bool) : A bool that specifies if only labels shall be converted
    """
    def __init__(self, dataset, limit=-1, only_labels=False):
        assert isinstance(dataset, torch.utils.data.Dataset)
        self.dataset = dataset
        self.limit = len(self.dataset) if limit == -1 else min(limit, len(self.dataset))
        # SUPERVISED
        if type(dataset[0]) == tuple:
            # NORMAL MODE
            if not (type(dataset[0][0]) == tuple or type(dataset[0][1]) == tuple): 
                if only_labels == True:
                    _, self.labels = zip(*[(None, dataset[i][1]) if type(dataset[i][1]) == int or dataset[i][1].dim() == 0 else (None, torch.argmax(dataset[i][1])) for i in range(self.limit)])
                else:
                    self.data, self.labels = zip(*[(dataset[i][0], dataset[i][1]) if type(dataset[i][1]) == int or dataset[i][1].dim() == 0  else (dataset[i][0], torch.argmax(dataset[i][1])) for i in range(self.limit)])
                    self.data = torch.cat(self.data).reshape(self.limit, -1).numpy()
                self.labels = np.asarray(self.labels, dtype=np.float32)
            # TUPLE MODE
            else:
                if only_labels == False:
                # TUPLE INPUTS
                    if type(dataset[0][0]) == tuple:
                        self.data = []
                        n_inputs = len(dataset[0][0]) 
                        for t in range(n_inputs):
                            _d, _ = zip(*[(dataset[i][0][t], None) for i in range(self.limit)])
                            _d = torch.cat(_d).reshape(self.limit,-1).numpy()
                            self.data.append(_d)
                    else:
                        _d, _ = zip(*[(dataset[i][0], None) for i in range(self.limit)])
                        _d = torch.cat(_d).reshape(self.limit,-1).numpy()
                        self.data = _d
                # TUPLE OUTPUTS
                if type(dataset[0][1]) == tuple:
                    self.labels = []
                    n_labels = len(dataset[0][1])
                    for l in range(n_labels):
                        _, _l = zip(*[(None, dataset[i][1][l]) if type(dataset[i][1][l]) == int or dataset[i][1][l].dim() == 0 else (None, torch.argmax(dataset[i][1][l])) for i in range(self.limit)])
                        self.labels.append(np.asarray(_l, dtype=np.float32))
                else:
                    _, _l = zip(*[(None, dataset[i][1]) if type(dataset[i][1]) == int or dataset[i][1][l].dim() == 0 else (None, torch.argmax(dataset[i][1])) for i in range(self.limit)])
                    self.labels = np.asarray(_l, dtype=np.float32)
        # UNSUPERVISED
        else:
            self.data, _ = zip(*[(dataset[i], None) for i in range(self.limit)])
            self.data = torch.cat(self.data).reshape(self.limit, -1).numpy()

        
    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)
