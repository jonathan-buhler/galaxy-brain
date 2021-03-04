import math

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

from datasets import DATASET_PATH, HDG10
from utils import sample_images

N_EPOCHS = 200
SAMPLE_INTERVAL = 500
BATCH_SIZE = 128
G_LR = 0.00005
D_LR = 0.00005
GAMMA = 0.995
BETAS = (0.5, 0.999)

N_CLASSES = 4
N_CHANNELS = 3
IMG_SIZE = 69  # 207
LATENT_DIM = 100

IMG_SHAPE = (IMG_SIZE, IMG_SIZE, N_CHANNELS)
IMG_AREA = int(math.prod(IMG_SHAPE))


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(N_CLASSES, N_CLASSES)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(LATENT_DIM + N_CLASSES, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, IMG_AREA),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *IMG_SHAPE)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(N_CLASSES, N_CLASSES)

        self.model = nn.Sequential(
            nn.Linear(N_CLASSES + IMG_AREA, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity


# Loss functions
# adversarial_loss = nn.MSELoss()
adversarial_loss = nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

dataloader = DataLoader(HDG10(DATASET_PATH), batch_size=BATCH_SIZE, shuffle=True)

# Optimizers
optimizer_G = Adam(generator.parameters(), lr=G_LR, betas=BETAS)
optimizer_D = Adam(discriminator.parameters(), lr=D_LR, betas=BETAS)

# scheduler_G = ExponentialLR(optimizer_G, gamma=GAMMA)
# scheduler_D = ExponentialLR(optimizer_D, gamma=GAMMA)


for epoch in range(N_EPOCHS):
    for batch, (imgs, labels) in enumerate(dataloader):
        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = torch.ones(batch_size, 1)
        fake = torch.zeros(batch_size, 1)

        # -----------------
        #  Train Generator
        # -----------------
        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = torch.tensor(
            np.random.normal(0, 1, (batch_size, LATENT_DIM)), dtype=torch.long
        )
        gen_labels = torch.tensor(
            np.random.randint(0, N_CLASSES, batch_size), dtype=torch.long
        )

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()
        # scheduler_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        # scheduler_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, N_EPOCHS, batch, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + batch
        if batches_done != 0 and batches_done % SAMPLE_INTERVAL == 0:
            sample_images(
                generator=generator,
                latent_dim=LATENT_DIM,
                n_classes=N_CLASSES,
                run_name="nsched",
                batch_count=batches_done,
            )
