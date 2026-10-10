import torch
import matplotlib.pyplot as plt

from data import get_train_loader
from VAE import VAE                  
from vae_trainer import train_loops 


def main():
    torch.manual_seed(42)

    train_loader = get_train_loader(batch_size=64)

    model = VAE(
        in_channels=1,
        latent_channels=4,
        hidden_channels=(32, 64),
        var=1.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    
    images, _ = next(iter(train_loader))
    images = images[:8].clone()

    loss_history = train_loops(
        model,
        optimizer,
        train_loader,
        num_epochs=1,
        beta=1e-3,
    )

    
    model.eval()
    with torch.no_grad():
        x_mu, _, _ = model(images)

    if not torch.isfinite(x_mu).all():
        raise RuntimeError("重建结果包含 NaN 或 Inf，请检查训练损失。")

    print("原图形状：", images.shape)
    print("重建形状：", x_mu.shape)
    print(
        "这些训练样本的重建 MSE：",
        (images - x_mu).square().mean().item(),
    )

  
    originals = ((images.cpu() + 1) / 2).clamp(0, 1)
    reconstructions = ((x_mu.cpu() + 1) / 2).clamp(0, 1)

    fig, axes = plt.subplots(2, len(images), figsize=(12, 3.5))
    for i in range(len(images)):
        axes[0, i].imshow(
            originals[i, 0].numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[1, i].imshow(
            reconstructions[i, 0].numpy(), cmap="gray", vmin=0, vmax=1
        )
        axes[0, i].axis("off")
        axes[1, i].axis("off")

    fig.suptitle("Top: Original | Bottom: VAE reconstruction")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(7, 3))
    plt.plot(loss_history)
    plt.xlabel("Training step")
    plt.ylabel("Total VAE loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()