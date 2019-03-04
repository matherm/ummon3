from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings

class MergeDataset(Dataset):
    """
    Merges a dataset into one by combining the flattened features.

    Example
    =======

    ds1 = NumpyDataset(data1) # shape x=(B, 200), y=(B, 1)
    ds2 = NumpyDataset(data2) # shape x=(B, 300), y=(B, 10)

    dsm = MergeDataset(ds1, ds2, label_idx=0) # shape x=(B, 500), y=(B, 1)

    Arguments
    ==========
    dataset_a (Dataset) : The first dataset
    dataset_b (Dataset) : The second dataset
    label_idx (int)     : which label vector to use
    """
    def __init__(self, dataset_a, dataset_b, label_idx=0):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.label_idx = label_idx
        assert len(dataset_a) == len(dataset_b)
        assert label_idx == 0 or label_idx == 1
        
    def __len__(self):
        return len(self.dataset_a)

    def __getitem__(self, idx):
        if self.label_idx == 0:
            label = self.dataset_a[idx][1] 
        else:
            label = self.dataset_b[idx][1]  
        data_l = self.dataset_a[idx][0]
        data_r = self.dataset_b[idx][0]
        # concatenate along feature axis ((,2), (,2) ==> (,4))
        data = torch.cat((data_l, data_r), 0) 
        return data, label