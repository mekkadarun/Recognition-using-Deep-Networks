import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np
from train_dcgan import Generator, Discriminator


def visualize_gan_results(generator, fixed_noise, real_images=None, g_losses=None, d_losses=None):
    """
    Visualize results from a trained GAN model

    Parameters:
    - generator: Trained generator model
    - fixed_noise: Noise vector used to generate images
    - real_images: Optional real images for comparison
    - g_losses: Optional generator losses for plotting
    - d_losses: Optional discriminator losses for plotting
    """
    # Generate fake images
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()

    # Plot generated images
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1)
    plt.title("Generated Images")
    plt.imshow(vutils.make_grid(fake_images, padding=2, normalize=True).permute(1, 2, 0))
    plt.axis('off')

    # Plot real images if provided
    if real_images is not None:
        plt.subplot(2, 2, 2)
        plt.title("Real Images")
        plt.imshow(vutils.make_grid(real_images, padding=2, normalize=True).permute(1, 2, 0))
        plt.axis('off')

    # Plot Generator and Discriminator Losses
    if g_losses is not None and d_losses is not None:
        plt.subplot(2, 2, 3)
        plt.title("Generator Loss")
        plt.plot(g_losses)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

        plt.subplot(2, 2, 4)
        plt.title("Discriminator Loss")
        plt.plot(d_losses)
        plt.xlabel("Iterations")
        plt.ylabel("Loss")

    plt.tight_layout()
    plt.show()


def load_gan_models(generator_path, discriminator_path, generator_class, discriminator_class, num_gpu):
    """
    Load pretrained GAN models

    Parameters:
    - generator_path: Path to saved generator model
    - discriminator_path: Path to saved discriminator model
    - generator_class: Generator model class
    - discriminator_class: Discriminator model class

    Returns:
    - Loaded generator and discriminator models
    """
    # Initialize models
    generator = generator_class(num_gpu)
    discriminator = discriminator_class(num_gpu)

    # Load state dicts
    generator.load_state_dict(torch.load(generator_path))
    discriminator.load_state_dict(torch.load(discriminator_path))

    # Set to evaluation mode
    generator.eval()
    discriminator.eval()

    return generator, discriminator


# Example usage
def main():
    # Configure these based on your specific model
    CONFIG = {
        'latent_vector_size': 100,  # Size of the noise vector
        'num_gpus': 0,  # Number of GPUs
        'generator_path': '../trained_models/gan_models_generator.pth',
        'discriminator_path': '../trained_models/gan_models_discriminator.pth'
    }

    # Load models
    generator, discriminator = load_gan_models(
        CONFIG['generator_path'],
        CONFIG['discriminator_path'],
        Generator,  # Your generator model class
        Discriminator,  # Your discriminator model class
        num_gpu=CONFIG['num_gpus']
    )

    # Create fixed noise
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fixed_noise = torch.randn(64, CONFIG['latent_vector_size'], 1, 1, device=device)
    generator = generator.to(device)
    # Visualize results
    visualize_gan_results(
        generator,
        fixed_noise,
    )


if __name__ == '__main__':
    main()