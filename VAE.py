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
    def __init__():
        super().__init__()

    def forward(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(mu)
        return mu + eps * std

#decoder reconstruction
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

    def fprward(self):
        return