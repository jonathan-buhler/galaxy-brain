import os

import torch
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


def sample_real(dataloader, batch_size, run_name):
    run_path = f"./src/samples/{run_name}"

    os.makedirs(run_path, exist_ok=True)

    imgs = next(iter(dataloader))

    save_image(
        imgs,
        f"{run_path}/real_sample.jpg",
        nrow=batch_size // 8,
        normalize=True,
    )
