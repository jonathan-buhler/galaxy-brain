import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# DATASET_PATH = "./src/datasets/HDG10.h5"
DATASET_PATH = "./src/datasets/G10.h5"


class HDG10(Dataset):
    def __init__(self, path, transform=None):
        super(HDG10).__init__()

        self.path = path
        self.transform = transform

        # Number of images temporarily capped at 1024 for performance reasons
        with h5py.File(path, "r") as file:
            self.images = torch.tensor(np.array(file["images"]))
            self.labels = torch.tensor(np.array(file["ans"]), dtype=torch.long)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)
