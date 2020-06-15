import torch
from torch.utils import data


def _to_torch(x):
    if isinstance(x, torch.Tensor):
        return x
    else:  # Assume numpy ndarray
        return torch.from_numpy(x)


class DatasetWrapper(data.Dataset):

    def __init__(self, X, A, Y, indices):
        self.X = _to_torch(X)
        self.A = _to_torch(A)
        self.Y = _to_torch(Y)
        self.indices = indices

    def __len__(self):
        return 1

    def to(self, device):
        self.X = self.X.to(device)
        self.A = self.A.to(device)
        self.Y = self.Y.to(device)
        return self

    def __getitem__(self, idx):
        return self.X, self.A, self.Y, self.indices
