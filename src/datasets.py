from PIL import Image

import h5py
import numpy as np
from torch.utils import data


class HDG10(data.Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        self.transform = transform

        with h5py.File(path, "r") as file:
            self.images = np.array(file["images"])
            self.labels = np.array(file["ans"])

    def __getitem__(self, index):
        pixels = self.images[index]

        image = Image.fromarray(pixels, mode="RGB")
        label = self.labels[index]

        if self.transform:
            image = self.transform(image)

        return image, label

    def __len__(self):
        return len(self.images)
