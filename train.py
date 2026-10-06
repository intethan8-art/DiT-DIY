import torch
import data
import model
import flow
from trainer import train_loops
import sampler 
import matplotlib.pyplot as plt


# main script for training
train_loader = data.get_train_loader(64)
net = model.DIT(1, 28, 28, 4, 64, 128, 4, 256, 2)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_history = train_loops(net, optimizer, train_loader, 5, flow.make_score_pair)

# generate
noise = torch.randn(64, 1, 28, 28)
samples = sampler.sample_ode(net, noise, 100)

# plot
images = (samples.detach().cpu() + 1) / 2
images = images.clamp(0, 1)

fig, axes = plt.subplots(8, 8, figsize=(8, 8))
for ax, image in zip(axes.flat, images):
    ax.imshow(image[0].numpy(), cmap="gray", vmin=0, vmax=1)
    ax.axis("off")

plt.tight_layout()
plt.show()