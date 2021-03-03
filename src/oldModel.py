# import numpy as np
# import torch
# import torch.nn as nn
# from torch.autograd import Variable


# from torch.utils.data import DataLoader
# from datasets import HDG10


# N_EPOCHS = 200
# N_CLASSES = 10
# LATENT_DIM = 100
# IMG_SIZE = 207
# CHANNELS = 3
# LR = 0.0002
# BETA_1 = 0.5
# BETA_2 = 0.999


# def weights_init_normal(m):
#     classname = m.__class__.__name__
#     if classname.find("Conv") != -1:
#         torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find("BatchNorm2d") != -1:
#         torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#         torch.nn.init.constant_(m.bias.data, 0.0)


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         self.label_emb = nn.Embedding(N_CLASSES, LATENT_DIM)

#         # Initial size before upsampling
#         self.init_size = IMG_SIZE // 4
#         self.l1 = nn.Sequential(nn.Linear(LATENT_DIM, 128 * self.init_size ** 2))

#         self.conv_blocks = nn.Sequential(
#             nn.BatchNorm2d(128),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 128, 3, stride=1, padding=1),
#             nn.BatchNorm2d(128, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(128, 64, 3, stride=1, padding=1),
#             nn.BatchNorm2d(64, 0.8),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, CHANNELS, 3, stride=1, padding=1),
#             nn.Tanh(),
#         )

#     def forward(self, noise, labels):
#         gen_input = torch.mul(self.label_emb(labels), noise)
#         out = self.l1(gen_input)
#         out = out.view(out.shape[0], 128, self.init_size, self.init_size)
#         img = self.conv_blocks(out)
#         return img


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         def discriminator_block(in_filters, out_filters, bn=True):
#             """Returns layers of each discriminator block"""
#             block = [
#                 nn.Conv2d(in_filters, out_filters, 3, 2, 1),
#                 nn.LeakyReLU(0.2, inplace=True),
#                 nn.Dropout2d(0.25),
#             ]
#             if bn:
#                 block.append(nn.BatchNorm2d(out_filters, 0.8))
#             return block

#         self.conv_blocks = nn.Sequential(
#             *discriminator_block(CHANNELS, 16, bn=False),
#             *discriminator_block(16, 32),
#             *discriminator_block(32, 64),
#             *discriminator_block(64, 128),
#         )

#         # The height and width of downsampled image
#         ds_size = IMG_SIZE // 2 ** 4

#         # Output layers
#         self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())
#         self.aux_layer = nn.Sequential(
#             nn.Linear(128 * ds_size ** 2, N_CLASSES), nn.Softmax()
#         )

#     def forward(self, img):
#         print("HERE")
#         print(img.shape)
#         out = self.conv_blocks(img)
#         print("AND HERE")
#         print(out.shape)
#         out = out.view(out.shape[0], -1)
#         print("WHAT ABOUT HERE?")
#         print(out.shape)
#         validity = self.adv_layer(out)
#         print("OR HERE?")
#         label = self.aux_layer(out)
#         print("I BET WE DON'T GET HERE")
#         return validity, label


# # Loss functions
# adversarial_loss = torch.nn.BCELoss()
# auxiliary_loss = torch.nn.CrossEntropyLoss()

# # Initialize generator and discriminator
# generator = Generator()
# discriminator = Discriminator()

# # Initialize weights
# generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# # Optimizers
# optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR, betas=(BETA_1, BETA_2))
# optimizer_D = torch.optim.Adam(
#     discriminator.parameters(), lr=LR, betas=(BETA_1, BETA_2)
# )

# FloatTensor = torch.FloatTensor
# LongTensor = torch.LongTensor


# # ----------
# #  Training
# # ----------
# def train(dataloader):
#     for epoch in range(N_EPOCHS):
#         for i, (imgs, labels) in enumerate(dataloader):
#             batch_size = imgs.shape[0]

#             # Adversarial ground truths
#             valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
#             fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

#             # Configure input
#             real_imgs = Variable(imgs.type(FloatTensor))
#             labels = Variable(labels.type(LongTensor))

#             # -----------------
#             #  Train Generator
#             # -----------------

#             optimizer_G.zero_grad()

#             # Sample noise and labels as generator input
#             z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, LATENT_DIM))))
#             gen_labels = Variable(
#                 LongTensor(np.random.randint(0, N_CLASSES, batch_size))
#             )

#             # Generate a batch of images
#             gen_imgs = generator(z, gen_labels)
#             print(gen_imgs[0].shape)

#             # Loss measures generator's ability to fool the discriminator
#             validity, pred_label = discriminator(gen_imgs)
#             g_loss = 0.5 * (
#                 adversarial_loss(validity, valid)
#                 + auxiliary_loss(pred_label, gen_labels)
#             )

#             g_loss.backward()
#             optimizer_G.step()

#             # ---------------------
#             #  Train Discriminator
#             # ---------------------

#             optimizer_D.zero_grad()

#             # Loss for real images
#             real_pred, real_aux = discriminator(real_imgs)
#             d_real_loss = (
#                 adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)
#             ) / 2

#             # Loss for fake images
#             fake_pred, fake_aux = discriminator(gen_imgs.detach())
#             d_fake_loss = (
#                 adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)
#             ) / 2

#             # Total discriminator loss
#             d_loss = (d_real_loss + d_fake_loss) / 2

#             # Calculate discriminator accuracy
#             pred = np.concatenate(
#                 [real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0
#             )
#             gt = np.concatenate(
#                 [labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0
#             )
#             d_acc = np.mean(np.argmax(pred, axis=1) == gt)

#             d_loss.backward()
#             optimizer_D.step()

#             print("step")


# dataset = HDG10("./src/datasets/HDG10.h5")
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
# train(dataloader)
