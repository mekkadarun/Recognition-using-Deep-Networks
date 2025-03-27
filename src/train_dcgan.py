import os
import random
import timeit
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib
matplotlib.use("TkAgg")  # Choose appropriate backend

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset
from PIL import Image
from IPython.display import HTML

# Consolidated Configuration
CONFIG = {
    'dataroot': '/Users/yuyangtian/Downloads/celeba',
    'workers': 2,
    'batch_size': 128,
    'image_size': 64,
    'num_channels': 3,
    'latent_vector_size': 100,
    'generator_features': 64,
    'discriminator_features': 64,
    'num_epochs': 5,
    'learning_rate': 0.0002,
    'beta1': 0.5,
    'num_gpus': 0,
    'real_label': 1.,
    'fake_label': 0.,
    'save_path': '../trained_models'
}


"""Custom dataset for loading celebrity images with labels. e.g.(image_001, 1)"""
class CelebrityDataset(Dataset):

    def __init__(self, img_dir, label_file, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []


        with open(label_file, 'r') as f:
            for line in f:
                filename, class_id_str = line.strip().split()
                class_id = int(class_id_str)
                filepath = os.path.join(img_dir, filename)

                if os.path.isfile(filepath):
                    self.samples.append((filepath, class_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, class_id = self.samples[idx]

        img = Image.open(filepath).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, class_id

def set_random_seed(seed=999):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def weights_init(m):
    """Custom weights initialization for network layers."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


"""GAN Generator Network.
 converting z to data-space means ultimately creating a RGB image with the same size as the training images (i.e. 3x64x64)
"""
class Generator(nn.Module):

    def __init__(self, num_gpus):
        super().__init__()
        self.num_gpus = num_gpus
        self.main = nn.Sequential(
            # First layer: Transform latent vector to initial feature map
            # Input: Latent vector of size CONFIG['latent_vector_size']
            # Output: (batch_size, generator_features * 8, 4, 4)
            nn.ConvTranspose2d(CONFIG['latent_vector_size'],
                               CONFIG['generator_features'] * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(CONFIG['generator_features'] * 8),
            nn.ReLU(True),

            # Second layer: Upsample and reduce feature channels
            # Input: (batch_size, generator_features * 8, 4, 4)
            # Output: (batch_size, generator_features * 4, 8, 8)
            nn.ConvTranspose2d(CONFIG['generator_features'] * 8,
                               CONFIG['generator_features'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG['generator_features'] * 4),
            nn.ReLU(True),

            # Input: (batch_size, generator_features * 4, 8, 8)
            # Output: (batch_size, generator_features * 2, 16, 16)
            nn.ConvTranspose2d(CONFIG['generator_features'] * 4,
                               CONFIG['generator_features'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG['generator_features'] * 2),
            nn.ReLU(True),
            # Input: (batch_size, generator_features * 2, 16, 16)
            # Output: (batch_size, generator_features, 32, 32)
            nn.ConvTranspose2d(CONFIG['generator_features'] * 2,
                               CONFIG['generator_features'], 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG['generator_features']),
            nn.ReLU(True),

            nn.ConvTranspose2d(CONFIG['generator_features'],
                               CONFIG['num_channels'], 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


"""GAN Discriminator Network.
D, is a binary classification network that takes an image as input and outputs a scalar probability that the input image is real (as opposed to fake).
D takes a 3x64x64 input image, processes it through a series of Conv2d, BatchNorm2d, and LeakyReLU layers, 
and outputs the final probability through a Sigmoid activation function. 
"""
class Discriminator(nn.Module):

    def __init__(self, num_gpus):
        super().__init__()
        self.num_gpus = num_gpus
        self.main = nn.Sequential(
            # Input: (num_channels) x 64 x 64
            nn.Conv2d(CONFIG['num_channels'], CONFIG['discriminator_features'], 4, 2, 1, bias=False), # 32x32
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf) x 32 x 32
            nn.Conv2d(CONFIG['discriminator_features'],
                      CONFIG['discriminator_features'] * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG['discriminator_features'] * 2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*2) x 16 x 16
            nn.Conv2d(CONFIG['discriminator_features'] * 2,
                      CONFIG['discriminator_features'] * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG['discriminator_features'] * 4),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*4) x 8 x 8
            nn.Conv2d(CONFIG['discriminator_features'] * 4,
                      CONFIG['discriminator_features'] * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(CONFIG['discriminator_features'] * 8),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf*8) x 4 x 4
            nn.Conv2d(CONFIG['discriminator_features'] * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


def plot_training_losses(G_losses, D_losses):
    """
    Plot generator and discriminator losses during training.

    Args:
        G_losses (list): Generator losses throughout training
        D_losses (list): Discriminator losses throughout training

    Returns:
        matplotlib.figure.Figure: Loss plot figure
    """
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(D_losses, label="Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
# Create an animation of generated images.
def show_ani(img_list):
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    ims = [[plt.imshow(np.transpose(i, (1, 2, 0)), animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

    HTML(ani.to_jshtml())

#  Plot real and generated images side by side.
def plot_real_vs_generated_images(real_batch, img_list, device):
    """
    Args:
        real_batch (torch.Tensor): Batch of real images
        img_list (list): List of generated image grids
        device (torch.device): Device to move tensors to
    """
    plt.figure(figsize=(15, 15))

    # Real Images Subplot
    plt.subplot(1, 2, 1)
    plt.axis("off")
    plt.title("Real Images")
    plt.imshow(np.transpose(
        vutils.make_grid(
            real_batch[0].to(device)[:64],
            padding=5,
            normalize=True
        ).cpu(),
        (1, 2, 0)
    ))

    # Fake Images Subplot
    plt.subplot(1, 2, 2)
    plt.axis("off")
    plt.title("Generated Images")
    plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
    plt.show()

def train_gan():
    """Main GAN training loop."""
    # Determine device
    device = torch.device("cuda:0" if torch.cuda.is_available() and CONFIG['num_gpus'] > 0 else "cpu")
    print(f"Using device: {device}")

    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.CenterCrop(CONFIG['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = CelebrityDataset(
        img_dir='../data/celeba_10k',
        label_file='../data/CelebA_Identity.txt',
        transform=transform
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['workers']
    )

    # Initialize networks
    netG = Generator(CONFIG['num_gpus']).to(device)
    netD = Discriminator(CONFIG['num_gpus']).to(device)

    # Handle multi-GPU
    if device.type == 'cuda' and CONFIG['num_gpus'] > 1:
        netG = nn.DataParallel(netG, list(range(CONFIG['num_gpus'])))
        netD = nn.DataParallel(netD, list(range(CONFIG['num_gpus'])))

    # Initialize weights
    netG.apply(weights_init)
    netD.apply(weights_init)

    # Loss and optimizers
    criterion = nn.BCELoss()

    # Fixed noise for visualization
    fixed_noise = torch.randn(64, CONFIG['latent_vector_size'], 1, 1, device=device)

    # Optimizers
    optimizerD = optim.Adam(netD.parameters(),
                            lr=CONFIG['learning_rate'],
                            betas=(CONFIG['beta1'], 0.999))
    optimizerG = optim.Adam(netG.parameters(),
                            lr=CONFIG['learning_rate'],
                            betas=(CONFIG['beta1'], 0.999))

    # Training tracking
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    model_start = datetime.now().timestamp()
    print(f'\nTraining starts at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(model_start))}')
    print("Starting Training Loop...")
    # Training Loop
    for epoch in range(CONFIG['num_epochs']):
        # For each batch in the dataloader
        start = timeit.default_timer()
        for i, data in enumerate(dataloader, 0):
            # (1) Update Discriminator Network
            netD.zero_grad()

            # Real images
            # Format batch
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), CONFIG['real_label'], dtype=torch.float, device=device)

            # Forward pass real batch through D
            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            # Fake images
            noise = torch.randn(b_size, CONFIG['latent_vector_size'], 1, 1, device=device)
            fake = netG(noise)
            label.fill_(CONFIG['fake_label'])

            # Classify fake batch with D
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()

            # Total Discriminator Loss
            errD = errD_real + errD_fake
            optimizerD.step()

            # (2) Update Generator Network
            netG.zero_grad()
            label.fill_(CONFIG['real_label'])  # fake labels are real for generator cost

            # Forward pass fake batch through D
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Training Statistics
            if i % 50 == 0:
                print(f'[{epoch}/{CONFIG["num_epochs"]}][{i}/{len(dataloader)}]\t'
                      f'Loss_D: {errD.item():.4f}\tLoss_G: {errG.item():.4f}\t'
                      f'D(x): {D_x:.4f}\tD(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f}')

            # Save Losses for plotting
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # Save generated images
            if (iters % 500 == 0) or ((epoch == CONFIG['num_epochs'] - 1) and (i == len(dataloader) - 1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        end = timeit.default_timer()
        print(f'[{epoch + 1}/{CONFIG["num_epochs"]}] takes {end-start:.2f} seconds')
        print(f'[{epoch + 1}/{CONFIG["num_epochs"]}] takes {end - start:.2f} seconds')

    model_end = datetime.now().timestamp()
    print("Ending Training Loop...")
    print(f'\nTraining ends at {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(model_end))}')

    # Save models
    torch.save(netG.state_dict(), os.path.join(CONFIG['save_path'], 'gan_models_generator.pth'))
    torch.save(netD.state_dict(), os.path.join(CONFIG['save_path'], 'gan_models_discriminator.pth'))

    # Plot loss
    plot_training_losses(G_losses, D_losses)
    real_batch = next(iter(dataloader))
    # plot real vs generated
    plot_real_vs_generated_images(real_batch, img_list, device)
    # Show the generation progress
    show_ani(img_list)


    plt.close()

    return {
        'generator': netG,
        'discriminator': netD,
        'g_losses': G_losses,
        'd_losses': D_losses,
        'generated_images': img_list
    }


def main():
    """Main script entry point."""
    # Train GAN and get results
    # Set random seed for reproducibility
    set_random_seed()
    train_gan()


if __name__ == "__main__":
    main()