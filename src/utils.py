import torch
import numpy as np
from torchvision.utils import save_image
from PIL import Image


def sample_images(generator, latent_dim, n_classes, dir, filename):
    z = torch.tensor(
        np.random.normal(0, 1, (n_classes ** 2, latent_dim)), dtype=torch.long
    )
    labels = np.array([num for _ in range(n_classes) for num in range(n_classes)])
    labels = torch.tensor(labels, dtype=torch.long)

    gen_imgs = generator(z, labels)

    sanity = gen_imgs[0].detach().numpy()
    img = Image.fromarray(sanity, mode="RGB")
    img.save(f"./src/{dir}/{filename}-sanity.jpg")


    insanity = gen_imgs.detach().view(gen_imgs.size(0),3,69,69)
    save_image(insanity, f"./src/{dir}/{filename}.jpg", nrow=n_classes, normalize=True)
