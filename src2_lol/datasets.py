import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

DATASET_PATH = "./src2_lol/datasets/G10.h5"

SEED = 1234
torch.manual_seed(SEED)


class G10(Dataset):
    def __init__(self, img_size, use_labels=False, just_spirals=False):
        super(G10).__init__()

        self.use_labels = use_labels

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

        p = 0.5
        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                # transforms.RandomHorizontalFlip(p),
                # transforms.RandomApply([transforms.RandomAffine(90, (0.4, 0.4))], p),
                # transforms.RandomErasing(p, (0.1, 0.6)),
                # transforms.Normalize(self.mean, self.std),
                # transforms.Grayscale(),
            ]
        )

    def __getitem__(self, index):
        if self.use_labels:
            return self.transform(self.images[index]), self.labels[index]
        else:
            return self.transform(self.images[index])

    def __len__(self):
        return len(self.images)

images = []
dataset = G10(img_size=64, use_labels=True)
for c in range(10):
    for (image, label) in dataset:
        if images and len(images) >= c * 10:
            print(len(images))
            break
        elif label.item() == c:
            images.append(image)

from torchvision.utils import save_image
save_image(images, "./src2_lol/image_classes-n.jpg", nrow=10, normalize=True)
