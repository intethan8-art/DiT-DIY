import torch
import math
import data
import model
import flow
from VAE import VAE
from trainer import train_loops
import sampler 
import matplotlib.pyplot as plt

# load pretrained vae
checkpoint = torch.load(
    "outputs/vae.pt", map_location="cpu", weights_only=True
)

vae = VAE(**checkpoint["config"])
vae.load_state_dict(checkpoint["model_state_dict"])

vae.eval()
vae.requires_grad_(False)

# directly working in latent space
train_loader = data.get_train_loader(64)
net = model.DIT(4, 7, 7, 1, 64, 128, 4, 256, 2)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_history = train_loops(net, optimizer, train_loader, 5, flow.make_score_pair, vae)

# generate
def sigma_t(t):
    return 0.5 * math.cos(math.pi / 2 * t)

noise = torch.randn(64, 4, 7, 7)
labels = torch.full((64,), 3, device=noise.device, dtype=torch.long)
z_generated = sampler.sample_sde(net, noise, labels, 100, sigma_t, w=2)
with torch.no_grad():
    images_mu = vae.decoder(z_generated)
eps = torch.randn_like(images_mu)
samples = images_mu + eps * torch.sqrt(vae.var)

# plot
images = (samples.detach().cpu() + 1) / 2
images = images.clamp(0, 1)

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for ax, image in zip(axes.flat, images):
    ax.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()