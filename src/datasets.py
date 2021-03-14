import h5py
import numpy as np
import torch
import math
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.transforms import Grayscale

DATASET_PATH = "./src/datasets/G10.h5"

SEED = 1234
torch.manual_seed(SEED)


class G10(Dataset):
    def __init__(self, img_size, just_spirals=False):
        super(G10).__init__()

        with h5py.File(DATASET_PATH, "r") as file:
            self.images = torch.tensor(np.array(file["images"]), dtype=torch.float)
            self.images = self.images.permute(0, 3, 1, 2).contiguous()

            self.mean = (
                self.images[0:, :, :, :].mean(),
                self.images[1, :, :, :].mean(),
                self.images[2, :, :, :].mean(),
            )

            self.std = (
                self.images[0:, :, :, :].std(),
                self.images[1, :, :, :].std(),
                self.images[2, :, :, :].std(),
            )

            self.labels = torch.tensor(np.array(file["ans"]), dtype=torch.long)

            if just_spirals:
                tight = self.labels == 7
                medium = self.labels == 8
                loose = self.labels == 9

                spirals = torch.logical_or(tight, torch.logical_or(medium, loose))

                self.images = self.images[spirals]
                self.labels = self.labels[spirals]

        p = 0.1
        augmentor = transforms.RandomApply([
            transforms.RandomAffine(0, (0, 0.125)),
            transforms.RandomAffine(0, None, (0.1, 0.6)),
            transforms.ColorJitter(),
            transforms.RandomRotation(90),
        ], p)


        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.Normalize(self.mean, self.std),
                transforms.RandomHorizontalFlip(p),
                # transforms.RandomRotation(90),
                augmentor,
                transforms.RandomErasing(p)
            ]
        )

    def __getitem__(self, index):
        # return self.transform(self.images[index]), self.labels[index]
        return self.transform(self.images[index])

    def __len__(self):
        return len(self.images)
