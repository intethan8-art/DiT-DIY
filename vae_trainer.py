import torch
import torch.nn.functional as F

def vae_loss(x, x_mu, mu, logvar, var, beta):
    recon = (x - x_mu).square().flatten(1).sum(dim=1)
    recon = recon / (2 * var)

    kl = 0.5 * (
        mu.square() + logvar.exp() - 1 - logvar
    ).flatten(1).sum(dim=1)

    loss = (recon + beta * kl).mean()
    return loss, recon.mean(), kl.mean()

def train_step(model, optimizer, x, beta):
    optimizer.zero_grad()
    x_mu, mu, logvar = model(x)
    loss, recon_loss, kl_loss = vae_loss(x, x_mu, mu, logvar, model.var, beta)
    loss.backward()
    optimizer.step()
    return loss.item(), recon_loss.item(), kl_loss.item()

def train_loops(model, optimizer, train_loader, num_epochs, beta):
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        sample_count = 0

        for step, (images, _) in enumerate(train_loader):
            loss, recon_loss, kl_loss = train_step(model, optimizer, images, beta)
            loss_history.append(loss)

            batch_size = images.shape[0]
            epoch_loss += loss * batch_size
            epoch_recon += recon_loss * batch_size
            epoch_kl += kl_loss * batch_size
            sample_count += batch_size

            if (step + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} "
                    f"Batch {step + 1}/{len(train_loader)} "
                    f"Mean loss: {epoch_loss / sample_count:.4f}"
                    f"Recon: {epoch_recon / sample_count:.4f} "
                    f"KL: {epoch_kl / sample_count:.4f}"
                )
    return loss_history