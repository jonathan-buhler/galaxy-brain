import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image

N_IMAGES = 64


def gen_samples(generator, latent_dim, run_name, batch_count):
    run_path = f"./src/samples/{run_name}"

    os.makedirs(run_path, exist_ok=True)

    # Due to budget-cuts we are no longer using labels/classes...
    noise = torch.randn(N_IMAGES, latent_dim, 1, 1)
    imgs = generator(noise)

    save_image(
        imgs,
        f"{run_path}/{batch_count}.jpg",
        nrow=N_IMAGES // 8,
        normalize=True,
    )
