import torch
import torch.nn as nn
import torch.nn.functional as F

# Build a VAE model, using for latent diffusion model.

#using convolution layers for encoder
class Encoder(nn.Module):
    def __init__(self, in_channels, latent_channels, hidden_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            hidden_channels[0],
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            hidden_channels[0], 
            hidden_channels[1],
            kernel_size=4, 
            stride=2, 
            padding=1
        )

        self.mu_head = nn.Conv2d(hidden_channels[1], latent_channels, kernel_size=1)
        self.logvar_head = nn.Conv2d(hidden_channels[1], latent_channels, kernel_size=1)

    def forward(self, x):
        x1 = F.gelu(self.conv1(x))
        x2 = F.gelu(self.conv2(x1))
        mu = self.mu_head(x2)
        logvar = self.logvar_head(x2)
        return mu, logvar

#reparameterize
class Reparameterize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(mu)
        return mu + eps * std

#decoder reconstruction
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, hidden_channels[0], kernel_size=1)

        self.conv2 = nn.ConvTranspose2d(
            hidden_channels[0],
            hidden_channels[1],
            kernel_size=4,
            stride=2,
            padding=1
        )

        self.conv3 = nn.ConvTranspose2d(
            hidden_channels[1],
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )

    def forward(self, z):
        z1 = F.gelu(self.conv1(z))
        z2 = F.gelu(self.conv2(z1))
        x_mu = F.tanh(self.conv3(z2))
        return x_mu

class VAE(nn.Module):
    def __init__(self, in_channels, latent_channels, hidden_channels, var):
        super().__init__()
        self.encoder = Encoder(in_channels, latent_channels, hidden_channels)
        self.reparameterize = Reparameterize()
        self.decoder = Decoder(latent_channels, in_channels, (hidden_channels[1], hidden_channels[0]))
        self.var = var

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_mu = self.decoder(z)
        return x_mu, mu, logvar