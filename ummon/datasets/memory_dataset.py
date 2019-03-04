from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings
import torch

class LoadDatasetIntoRam(Dataset):
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
        self.data = []
        self.labels = []
        # SUPERVISED
        if type(dataset[0]) == tuple:
            # NORMAL MODE
            if not (type(dataset[0][0]) == tuple or type(dataset[0][1]) == tuple): 
                if only_labels == True:
                    _, labels = zip(*[(None, torch.tensor(dataset[i][1]).unsqueeze(0)) for i in range(self.limit)])
                else:
                    data, labels = zip(*[(dataset[i][0], torch.tensor(dataset[i][1]).unsqueeze(0)) for i in range(self.limit)])
                    self.data = torch.cat(data).reshape(self.limit, -1).numpy()
                self.labels = torch.cat(labels).reshape(self.limit, -1).numpy()
            # TUPLE MODE
            else:
                if only_labels == False:
                # TUPLE INPUTS
                    if type(dataset[0][0]) == tuple:
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
                    n_labels = len(dataset[0][1])
                    for l in range(n_labels):
                        _, _l = zip(*[(None,  torch.tensor(dataset[i][1][l]).unsqueeze(0)) for i in range(self.limit)])
                        _l = torch.cat(_l).reshape(self.limit, -1).numpy()
                        self.labels.append(_l)
                else:
                    _, _l = zip(*[(None, torch.tensor(dataset[i][1]).unsqueeze(0)) for i in range(self.limit)])
                    self.labels = torch.cat(_l).reshape(self.limit, -1).numpy()
       # UNSUPERVISED
        else:
            data, _ = zip(*[(dataset[i], None) for i in range(self.limit)])
            self.data = torch.cat(data).reshape(self.limit, -1).numpy()

    def get_data(self):
        return self.data, self.labels
        
    def __getitem__(self, idx):
        if type(self.data) == list:
            data = tuple([torch.from_numpy(dat[idx]) for dat in self.data])
        else:
            data = torch.from_numpy(self.data[idx])
        if type(self.labels) == list:
            labels = tuple([label[idx] for label in self.labels])
        else:
            labels = self.labels[idx]
        return data, labels 

    def __len__(self):
        return len(self.dataset)