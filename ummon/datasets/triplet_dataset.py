from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings

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