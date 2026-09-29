import torch
from model import MLP
from trainer import train_loops
from sampler import sample
import matplotlib.pyplot as plt


# main script for training
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_history = train_loops(model, optimizer, 500, 64)

# generate
noise = torch.randn(64, 2)
samples = sample(model, noise, 100)
points = samples.detach().cpu().numpy()

# plot
plt.scatter(points[:, 0], points[:, 1], s=10, alpha=0.5)
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(alpha=0.3)
plt.show()