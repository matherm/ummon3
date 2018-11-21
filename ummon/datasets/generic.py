from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings

__all__ = ["UnsupTensorDataset" , "SiameseTensorDataset", "SiameseDataset" , "WeightedPair", "Triplet", "TripletTensorDataset", "NumpyDataset"]

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
    
    It takes a dataset like cifar10 and ensures that balanced triples are returnd.
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
