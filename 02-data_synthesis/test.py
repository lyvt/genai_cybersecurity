import os
import torch
from model import Generator
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

# Configuration settings
cfg = {
    'gpu_usage': True,
    'seed': 42,
    'test_batch_size': 4,
    'save_dir': 'checkpoints',
    'image_size': 256,
    'label_dim': 9,
    'G_input_dim': 100,
    'G_output_dim': 3,
    'D_input_dim': 3,
    'D_output_dim': 3,
    'num_filters': [1024, 512, 256, 128, 64, 32]
}


def denorm(x):
    """De-normalization"""
    return x.clamp(0, 1).mul(0.5).add(0.5)  # Combine operations for efficiency


def test(cfg):
    """Test the model using the specified configuration."""
    device = torch.device('cuda' if cfg['gpu_usage'] else 'cpu')

    # Initialize the AutoEncoder model
    generator = Generator(cfg['G_input_dim'], cfg['label_dim'], cfg['num_filters'], cfg['G_output_dim']).to(device)

    # Load the saved model checkpoint
    generator_path = os.path.join(cfg['save_dir'], 'G_model_19.pth')
    if os.path.exists(generator_path):
        generator.load_state_dict(torch.load(generator_path, map_location=device))

    generator.eval()

    # Prepare noise and label
    onehot = torch.eye(cfg['label_dim']).view(cfg['label_dim'], cfg['label_dim'], 1, 1)  # One-hot encoding
    noise = torch.randn(cfg['test_batch_size'], cfg['G_input_dim'], 1, 1).to(device)

    c_ = (torch.rand(cfg['test_batch_size']) * cfg['label_dim']).long()
    label = onehot[c_].to(device)

    with torch.no_grad():  # Disable gradients for testing
        gen_image = generator(noise, label)
        gen_image = denorm(gen_image)

    n_samples = noise.size(0)
    n_rows = int(np.ceil(np.sqrt(n_samples)))
    n_cols = int(np.ceil(n_samples / n_rows))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5, 5))
    for ax, img in zip(axes.flatten(), gen_image):
        ax.axis('off')
        img = img.cpu().numpy().transpose(1, 2, 0)  # Move to CPU and transpose
        img = (img - img.min()) * 255 / (img.max() - img.min())  # Scale to [0, 255]
        ax.imshow(img.astype(np.uint8), aspect='equal')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.title(f'Epoch {20}')
    plt.savefig('data_synthesis_testing.png')
    plt.show()


if __name__ == "__main__":
    test(cfg)
