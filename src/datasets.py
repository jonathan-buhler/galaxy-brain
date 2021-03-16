import h5py
import numpy as np
import torch
from PIL import Image
import math
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision
from torchvision.transforms.transforms import Grayscale

DATASET_PATH = "./src/datasets/G10.h5"

SEED = 1234
torch.manual_seed(SEED)


class G10(Dataset):
    def __init__(self, img_size, no_classes=True, just_spirals=False):
        super(G10).__init__()

        self.no_classes = no_classes

        with h5py.File(DATASET_PATH, "r") as file:
            self.images = torch.tensor(np.array(file["images"]), dtype=torch.float)
            self.images = self.images.permute(0, 3, 1, 2).contiguous()

            self.labels = torch.tensor(np.array(file["ans"]), dtype=torch.long)

        if not no_classes:
            classes = [self.labels == i for i in range(10)]

            smooth = torch.logical_or(
                classes[1], torch.logical_or(classes[2], classes[3])
            )
            disks = torch.logical_or(
                classes[0],
                torch.logical_or(
                    classes[7],
                    torch.logical_or(classes[8], classes[9]),
                ),
            )

            edges = torch.logical_or(
                classes[4], torch.logical_or(classes[5], classes[6])
            )

            self.labels[smooth] = 0
            self.labels[disks] = 1
            self.labels[edges] = 2

        elif just_spirals:
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

        # p = 0.3
        # augmentor = transforms.RandomApply([
        #     transforms.RandomAffine(0, (0, 0.4)),
        #     transforms.RandomAffine(0, None, (0.1, 0.8)),
        #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
        #     transforms.RandomRotation(90),
        # ], p)

        self.transform = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                # transforms.RandomHorizontalFlip(p),
                # transforms.RandomApply([transforms.RandomAffine(0, (0, 0.4))], p),
                # transforms.RandomApply([transforms.RandomAffine(0, None, (0.1, 0.8))], p),
                # -------------------------------------------------------------------
                # transforms.RandomApply([RandomTranslateWithReflect(9, img_size)], p),
                # -------------------------------------------------------------------
                # transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p),
                # transforms.RandomApply([transforms.RandomRotation(90)], p),
                transforms.RandomErasing(0.1),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def __getitem__(self, index):
        if self.no_classes:
            return self.transform(self.images[index])
        else:
            return self.transform(self.images[index]), self.labels[index]

    def __len__(self):
        return len(self.images)


# class RandomTranslateWithReflect:
#     '''
#     Translate image randomly

#     Translate vertically and horizontally by n pixels where
#     n is integer drawn uniformly independently for each axis
#     from [-max_translation, max_translation].

#     Fill the uncovered blank area with reflect padding.
#     '''

#     def __init__(self, max_translation, img_size):
#         self.max_translation = max_translation
#         self.img_size = img_size

#     def __call__(self, old_image):
#         old_image = transforms.F.to_pil_image(old_image)
#         xtranslation, ytranslation = np.random.randint(-self.max_translation,
#                                                        self.max_translation + 1,
#                                                        size=2)
#         xpad, ypad = abs(xtranslation), abs(ytranslation)
#         xsize, ysize = (self.img_size, self.img_size)

#         flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
#         flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
#         flipped_both = old_image.transpose(Image.ROTATE_180)

#         new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

#         new_image.paste(old_image, (xpad, ypad))

#         new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
#         new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

#         new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
#         new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

#         new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
#         new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
#         new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
#         new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

#         new_image = new_image.crop((xpad - xtranslation,
#                                     ypad - ytranslation,
#                                     xpad + xsize - xtranslation,
#                                     ypad + ysize - ytranslation))
#         new_image = transforms.F.to_tensor(new_image)
#         return new_image
