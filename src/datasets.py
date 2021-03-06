import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_PATH = "./src/datasets/G10.h5"


class G10(Dataset):
    def __init__(self, img_size, just_spirals=False):
        super(G10).__init__()

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        with h5py.File(DATASET_PATH, "r") as file:
            self.images = torch.tensor(np.array(file["images"]), dtype=torch.float)
            self.images = self.images.permute(0, 3, 1, 2).contiguous()

            self.labels = torch.tensor(np.array(file["ans"]), dtype=torch.long)

            if just_spirals:
                tight = self.labels == 7
                medium = self.labels == 8
                loose = self.labels == 9

                spirals = torch.logical_or(tight, torch.logical_or(medium, loose))

                self.images = self.images[spirals]
                self.labels = self.labels[spirals]

    def __getitem__(self, index):
        # return self.transform(self.images[index]), self.labels[index]
        return self.transform(self.images[index])

    def __len__(self):
        return len(self.images)
