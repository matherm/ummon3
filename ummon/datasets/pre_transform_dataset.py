import sys, os, time
import os.path as osp
from pathlib import Path
from zipfile  import ZipFile

import numpy as np
import torch

from torch_cluster import knn
from torch.utils.data import DataLoader, Dataset

class PreTransformDataset(Dataset):
    r''' Simple dataset wrapper for pre_transforming transformations like VGG-19.

    @author Matthias Hermann
    '''
    def __init__(self, dataset, pre_transform, pre_filter=None, workers=1, path="__ummoncache__/pretransformed"):

        self.dataset = dataset
        self.workers = workers
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        if hasattr(dataset, "transform"):
            if dataset.transform is not None:
                raise ValueError("Dataset also specifies transforms. Exiting..")
        
        # choose the right data partition to load
        sample = pre_transform(dataset[0][0]) + pre_transform(dataset[len(dataset)-1][0])
        self.str_pretransform = hash_tensor(sample)

        # check uniqueness
        #assert self.str_pretransform != hash_tensor(sample[0] + 0.0001)
        #assert self.str_pretransform != hash_tensor(pre_transform(dataset[1][0]))
        
        self.filename_data   = "{}_data.pt".format(os.path.join(path, self.str_pretransform))
        self.filename_labels = "{}_labels.pt".format(os.path.join(path, self.str_pretransform))

        accquire_lock()

        if not os.path.exists(self.filename_data) or not os.path.exists(self.filename_labels):
            data, labels = self.process(self.dataset, self.pre_transform, self.pre_filter)
            if not os.path.exists(path): os.makedirs(path)
            torch.save(data, self.filename_data)
            torch.save(labels, self.filename_labels)

        release_lock()

        self.data, self.labels = torch.load(self.filename_data), torch.load(self.filename_labels)

        assert len(self.dataset) == len(self.data)

    def __repr__(self):
        return "{}({}, pre_transform={})".format(self.__class__.__name__, repr(self.dataset), str(self.pre_transform).replace("\n", ""))
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return len(self.dataset)

    def process(self, dataset, pre_transform, pre_filter):
        data_list, label_list = [], []
        ds = self.dataset
        if self.workers > 1:

            def simple_collate(batch): 
                return batch

            class SimpleDataset(Dataset):

                def __init__(self, dataset, transforms, pre_filter):
                    self.dataset, self.transforms, self.pre_filter = dataset, transforms, pre_filter

                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    if self.pre_filter is None or self.pre_filter(item):
                        return self.transforms(item[0]), self.dataset[idx][1]
                    else:
                        return None
                
                def __len__(self):
                    return len(self.dataset)

            # Bug: https://github.com/pytorch/pytorch/issues/973
            torch.multiprocessing.set_sharing_strategy('file_system')
            # Repoint the runtime transforms to the pre_transforms so that they are computed on-load
            ds  = DataLoader(SimpleDataset(dataset, pre_transform, pre_filter), batch_size=self.workers, shuffle=False, num_workers=self.workers, collate_fn=simple_collate)
        for i, item in enumerate(ds):

            if self.workers > 1:
            # Case batch processing                
                for p_item in item:
                    if p_item is not None:
                        if not torch.is_tensor(p_item[0]):
                            data_list.append(torch.Tensor(p_item[0]))
                        else:
                            data_list.append(p_item[0])
                        if not torch.is_tensor(p_item[1]):
                            label_list.append(torch.Tensor(p_item[1]))
                        else:
                            label_list.append(p_item[1])
            else:
            # Case single processing
                if self.pre_filter is not None:
                    if self.pre_filter(item) is None:
                        continue

                if self.pre_transform is not None:
                    item_data = self.pre_transform(item[0])       
                    item_label = item[1]
                    if not torch.is_tensor(item_data):
                        data_list.append(torch.Tensor(item_data))
                    else:
                        data_list.append(item_data)
                    if not torch.is_tensor(item_label):
                        label_list.append(torch.Tensor(item_label))
                    else:
                        label_list.append(item_label)

        if isinstance(ds, DataLoader):
            torch.multiprocessing.set_sharing_strategy('file_descriptor')

        return torch.stack(data_list), torch.stack(label_list)

import hashlib
def props(cls):   
  return [i for i in cls.__dict__.keys() if i[:1] != '_']

def hash_obj(obj):
    hashes = []
    for key in props(obj):
        hash_key = hashlib.md5(key.encode('utf-8')).hexdigest()
        hash_val = hashlib.md5(str(getattr(obj, key)).encode('utf-8')).hexdigest()
        hashes.append(hash_key + hash_val)
    hash_output = hashlib.md5("".join(hashes).encode("utf-8")).hexdigest()
    return hash_output

def hash(obj):
    return hashlib.md5(obj).hexdigest()

def hash_tensor(obj):
    if isinstance(obj, np.ndarray):
        return hash(obj.tobytes())
    return hash(obj.numpy().tobytes())

def accquire_lock(LOCK = ".pre_processing_lock"):
    while os.path.exists(LOCK):
        time.sleep(2)
        print("Trying to accquire lock: ", LOCK)
    with open(LOCK, "w") as f:
        f.write("locking preprocessing...")

def release_lock(LOCK = ".pre_processing_lock"):
    os.remove(LOCK)
