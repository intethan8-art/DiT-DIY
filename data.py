import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# naive 2D Gaussian objective data
def sample_data(batch_size=64):
    mu1 = -2.0
    mu2 = 2.0
    sigma = 0.5
    p = 0.5

    noise = torch.randn(batch_size, 2) * sigma
    r = torch.rand(batch_size)
    x_centers = torch.where(r < p, mu1, mu2)
    y_centers = torch.zeros_like(x_centers)
    centers = torch.stack([x_centers, y_centers], dim=1)
    return centers + noise

# load data MNIST
def get_train_loader(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])

    dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    return train_loader