import os

import numpy as np
import torch
from PIL import Image
from torchvision.utils import save_image


def sample_images(generator, latent_dim, n_classes, run_name, batch_count):
    run_path = f"./src/samples/{run_name}"

    os.makedirs(run_path, exist_ok=True)

    z = torch.tensor(
        np.random.normal(0, 1, (n_classes ** 2, latent_dim)), dtype=torch.long
    )
    labels = np.array([num for _ in range(n_classes) for num in range(n_classes)])
    labels = torch.tensor(labels, dtype=torch.long)

    gen_imgs = generator(z, labels)

    sanity = gen_imgs[0].detach().numpy()
    img = Image.fromarray(sanity, mode="RGB")
    img.save(f"{run_path}/{batch_count}-sanity.jpg")

    insanity = gen_imgs.detach().view(gen_imgs.size(0), 3, 69, 69)
    save_image(
        insanity,
        f"{run_path}/{batch_count}.jpg",
        nrow=n_classes,
        normalize=True,
    )
