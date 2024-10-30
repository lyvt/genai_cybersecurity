import torch
from torch import nn


class VAE_CNN(nn.Module):
    def __init__(self):
        super(VAE_CNN, self).__init__()

        # Define the encoder network
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False),  # Input: 3 channels, Output: 16 channels
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),  # Downsample
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),  # Increase channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False),   # Downsample
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.relu = nn.ReLU()
        # Fully connected layers for latent space
        self.fc1 = nn.Linear(25 * 25 * 16, 1024)
        self.fc_bn1 = nn.BatchNorm1d(1024)
        self.fc_mu = nn.Linear(1024, 1024)  # Mean of the latent space
        self.fc_logvar = nn.Linear(1024, 1024)  # Log variance of the latent space

        # Fully connected layers for decoding (sampling)
        self.fc3 = nn.Linear(1024, 1024)
        self.fc_bn3 = nn.BatchNorm1d(1024)
        self.fc4 = nn.Linear(1024, 25 * 25 * 16)

        # Define the decoder network
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        )

    def encode(self, x):
        # Pass input through the encoder
        conv_out = self.encoder(x)
        conv_out = conv_out.view(-1, 25 * 25 * 16)  # Flatten the output

        # Pass through fully connected layers
        fc_out = self.relu(self.fc_bn1(self.fc1(conv_out)))
        mu = self.fc_mu(fc_out)  # Mean
        logvar = self.fc_logvar(fc_out)  # Log variance
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Reparameterization trick
        std = torch.exp(0.5 * logvar)  # Calculate standard deviation
        eps = torch.randn_like(std)  # Sample epsilon
        return mu + eps * std  # Return sampled latent vector

    def decode(self, z):
        # Pass the latent vector through the fully connected layers
        fc_out = self.relu(self.fc_bn3(self.fc3(z)))
        fc_out = self.fc4(fc_out).view(-1, 16, 25, 25)  # Reshape output to match decoder input

        # Pass through the decoder
        decoded = self.decoder(fc_out)
        return decoded.view(-1, 3, 100, 100)  # Return the final decoded image

    def forward(self, x):
        # Define the forward pass
        mu, logvar = self.encode(x)  # Encode the input
        z = self.reparameterize(mu, logvar)  # Sample from the latent space
        return self.decode(z), mu, logvar  # Decode and return the output, mu, and logvar
