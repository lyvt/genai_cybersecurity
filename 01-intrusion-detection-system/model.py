import torch.nn as nn
from math import sqrt


class AutoEncoder(nn.Module):
    def __init__(self, n_features):
        """
        Initialize the AutoEncoder model.

        :param n_features: Number of input features.
        """
        super(AutoEncoder, self).__init__()

        # Define the encoder part of the AutoEncoder
        self.encoder = nn.Sequential(
            nn.Linear(n_features, round(n_features * 0.75)),  # Encoder input layer
            nn.Tanh(),  # Activation function
            nn.Linear(round(n_features * 0.75), round(n_features * 0.5)),  # Encoder hidden layer
            nn.Tanh(),  # Activation function
            nn.Linear(round(n_features * 0.5), round(sqrt(n_features)) + 1),  # Latent space layer
        )

        # Define the decoder part of the AutoEncoder
        self.decoder = nn.Sequential(
            nn.Linear(round(sqrt(n_features)) + 1, round(n_features * 0.5)),  # Decoder input layer
            nn.Tanh(),  # Activation function
            nn.Linear(round(n_features * 0.5), round(n_features * 0.75)),  # Decoder hidden layer
            nn.Tanh(),  # Activation function
            nn.Linear(round(n_features * 0.75), n_features),  # Decoder output layer
        )

    def forward(self, x):
        """
        Forward pass through the AutoEncoder.

        :param x: Input tensor.
        :return: Tuple of (latent space representation, reconstructed input).
        """
        # Encode the input
        latent_z = self.encoder(x)

        # Decode the latent representation
        x_reconstruction = self.decoder(latent_z)

        return latent_z, x_reconstruction

