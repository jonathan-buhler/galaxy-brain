import h5py
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset


class HDG10(Dataset):
    def __init__(self, path, transform=None):
        super(HDG10).__init__()

        self.path = path
        self.transform = transform

        # Number of images temporarily capped at 1024 for performance reasons
        with h5py.File(path, "r") as file:
            self.images = Tensor(np.array(file["images"][:1024]))
            self.labels = Tensor(np.array(file["ans"][:1024]))

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
