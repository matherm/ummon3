from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings
import torch
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


class SiameseDataset(Dataset):
    """Dataset wrapping datasets for siamese networks.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        dataset_l (Dataset): contains sample data.
        dataset_ (Dataset): contains sample data.
    """

    def __init__(self, dataset_l, dataset_r):
        self.dataset_l = dataset_l
        self.dataset_r = dataset_r
        assert len(self.dataset_l) == len(self.dataset_r)

    def __repr__(self):
        return repr(self.dataset_l)

    def __getitem__(self, index):
        return ((self.dataset_l[index][0], self.dataset_r[index][0]) , (self.dataset_l[index][1], self.dataset_r[index][1]))

    def __len__(self):
        return len(self.dataset_l)