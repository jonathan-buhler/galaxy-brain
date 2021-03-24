from matplotlib import pyplot as plt
from torchvision.utils import save_image


def sample_real(dataloader, batch_size, save_path, labled=False):
    imgs = None
    if labled:
        imgs, _ = next(iter(dataloader))
    else:
        imgs = next(iter(dataloader))

    save_image(
        imgs,
        f"{save_path}/real_sample.jpg",
        nrow=batch_size // 8,
        normalize=True,
    )


def plot_loss(g_losses, d_losses, model_name, save_path):
    plt.plot(range(len(g_losses)), g_losses, label="Generator")
    plt.plot(range(len(d_losses)), d_losses, label="Discriminator")

    plt.title(f"{model_name} Mean Training Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Loss")

    plt.legend()

    plt.savefig(f"{save_path}/losses.jpg")
