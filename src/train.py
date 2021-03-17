import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from datasets import G10
from ganaxy import Discriminator_Stage1, Generator_Stage1
from utils import sample_real

N_EPOCHS = 200
BATCH_SIZE = 64

LATENT_DIM = 100
IMG_SIZE = 64

LR = 0.0002
BETAS = (0.5, 0.999)

SAMPLE_INTERVAL = 20
DIR = "rev-ganaxy"
DIR_PATH = f"./src/samples/{DIR}"
os.makedirs(DIR_PATH, exist_ok=True)

CUDA = True if torch.cuda.is_available() else False


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


generator = Generator_Stage1()
discriminator = Discriminator_Stage1()

adversarial_loss = nn.BCELoss()

if CUDA:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
dataset = G10(img_size=IMG_SIZE, just_spirals=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

sample_real(dataloader=dataloader, batch_size=BATCH_SIZE, run_name=DIR)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=BETAS)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=BETAS)

Tensor = torch.cuda.FloatTensor if CUDA else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(N_EPOCHS):
    for i, imgs in enumerate(dataloader):
        # Adversarial ground truths
        # valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        # fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)
        valid = torch.ones(imgs.shape[0])
        fake = torch.zeros(imgs.shape[0])

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        # z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM))))
        z = Variable(torch.randn(imgs.shape[0], LATENT_DIM, 1, 1))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, N_EPOCHS, i, len(dataloader), d_loss.item(), g_loss.item())
        )

        batches_done = epoch * len(dataloader) + i
        if batches_done % SAMPLE_INTERVAL == 0:
            save_image(
                gen_imgs.data[:25],
                f"{DIR_PATH}/{batches_done}.jpg",
                nrow=5,
                normalize=True,
            )
