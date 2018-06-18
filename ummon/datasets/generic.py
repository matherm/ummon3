from torch.utils.data import Dataset
from torch.utils.data import DataLoader

__all__ = ["UnsupTensorDataset" , "SiameseTensorDataset" , "WeightedPair", "ImagePatches", "AnomalyImagePatches", "TripletTensorDataset"]

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
            while str(self.class_ratio) == "0.0" and rnd_idx in self.class_idx[left_label]:
                rnd_idx = np.random.randint(len(self.labels), size=1)[0]

            right_img, right_label = self.dataset[rnd_idx]

        return (left_img, right_img), (left_label, right_label)

    def __len__(self):
        return len(self.dataset)


from torchvision import transforms
from imageio import imread
class ImagePatches(Dataset):
    """
    Dataset for generating data from a single given image. It used a window-scheme, hence the name ImageTiles.
    
    Arguments:
        * file (str) : The image filename
        * mode (str) : The processing mode 'bgr' or 'gray' (default="bgr")
        * train (bool) : train or test set
        * train_percentage (float) : percentage of train patches compared to all patches
        * transform (torchvision.transforms) : Image Transformations (default transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        * stride_x : The overlap in x direction
        * stride_y : The overlap in y direction
        * window_size (int) : square size of the resulting patches
    
    """
    
    def __init__(self, file, mode='bgr', train = True, train_percentage=.8, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                 stride_x=16, stride_y=16, window_size=32):

        self.filename = file
        self.img = imread(file)
        self.train = train
        self.train_percentage = train_percentage
        self.stride_x = stride_x
        self.stride_y = stride_y
        self.window_size = window_size
        self.transform = transform
        self.dataset_size = int(((self.img.shape[0] - self.window_size) / self.stride_y) + 1) * int(((self.img.shape[1] - self.window_size) / self.stride_x) + 1)

        if mode == 'bgr':
            self.__rgb_to_bgr__()
        elif mode == 'gray':
            self.__rgb_to_gray__()

    def __rgb_to_bgr__(self):
        r = np.expand_dims(self.img[:, :, 0], axis=2)
        g = np.expand_dims(self.img[:, :, 1], axis=2)
        b = np.expand_dims(self.img[:, :, 2], axis=2)
        self.img = np.concatenate((b,g,r), axis=2)
        

    def __rgb_to_gray__(self):
        r = np.expand_dims(self.img[:, :, 0], axis=2)
        g = np.expand_dims(self.img[:, :, 1], axis=2)
        b = np.expand_dims(self.img[:, :, 2], axis=2)
        self.img = (.2989 * r) + (.5870 * g) + (.114 * b)
        
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

    def _get_patch(self, idx):
        y = idx // (((self.img.shape[0] - self.window_size) // self.stride_y) + 1)
        x = idx % (((self.img.shape[1] - self.window_size) // self.stride_x) + 1)

        topleft_y = y * self.stride_y
        bottomright_y = y * self.stride_y + self.window_size
        topleft_x = x * self.stride_x
        bottomright_x = x * self.stride_x + self.window_size

        patch = self.img[topleft_y : bottomright_y, topleft_x : bottomright_x, :]
        
        # normalize
        patch = patch.astype(np.float32)
        if patch.max() > 1:
            patch = patch / 255.
            
        return patch

    def __getitem__(self, idx):
        
        patch = self._get_patch(idx)

        if self.transform:
            return self.transform(patch), 1
        return patch, 1
    
from ummon.preprocessing.anomaly import *
class AnomalyImagePatches(ImagePatches):
     """
     
     Dataset for generating data and anomaly from a single given image. It used a window-scheme, hence the name ImageTiles.
     
     Note
     ====
     Compared to ImagePatches all anomaly patches have label -1 whereas non-anomaly patches have 1
    
     Arguments:
        * file (str) : The image filename
        * mode (str) : The processing mode 'bgr' or 'gray' (default="bgr")
        * train (bool) : train or test set
        * train_percentage (float) : percentage of train patches compared to all patches
        * transform (torchvision.transforms) : Image Transformations (default transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        * stride_x : The overlap in x direction
        * stride_y : The overlap in y direction
        * window_size (int) : square size of the resulting patches
        * anomaly (torchvision.transformation)
        * propability (float) : propability of a anomaly
    
     """
     def __init__(self, file, mode='bgr', train = True, train_percentage=.8, transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
                 stride_x=16, stride_y=16, window_size=32, anomaly=SquareAnomaly(), propability=0.2):
        super(AnomalyImagePatches, self).__init__(file, mode, train, train_percentage, transform, stride_x, stride_y, window_size)

        self.anomaly = anomaly
        self.propability = propability

     def __getitem__(self, idx):
        
        # Get the patch
        patch = self._get_patch(idx)
        
        # Add anomaly with propability given by self.propability and label -1 respectivly
        if np.random.rand() < self.propability:
            # swap channel to front
            if patch.ndim == 3 and patch.shape[0] > patch.shape[2]:
                patch = np.transpose(patch, (2,0,1))
            patch = torch.from_numpy(patch)
            patch = self.anomaly(patch)
            patch = patch.numpy()
            label = -1
        else:
            label = 1

        # Apply transformations
        if self.transform:
            return self.transform(patch), label
        return patch, label
    
    
    
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