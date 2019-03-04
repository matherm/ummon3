from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import warnings

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
        return self.data_tensor[index], self.data_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)