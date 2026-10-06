import torch
import torch.nn.functional as F
from data import sample_data

# training process for one batch
def train_step(model, optimizer, z, make_pair):
    optimizer.zero_grad()
    x_t, t, u_target = make_pair(z)
    u_pred = model(x_t, t)
    loss = F.mse_loss(u_pred, u_target)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_loops(model, optimizer, num_steps, batch_size, make_pair):
    loss_history = []
    for _ in range(num_steps):
        z = sample_data(batch_size) 
        loss = train_step(model, optimizer, z, make_pair)
        loss_history.append(loss)
    return loss_history

def train_loops(model, optimizer, train_loader, num_epochs, make_pair):
    loss_history = []
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        sample_count = 0

        for step, (images, _) in enumerate(train_loader):
            loss = train_step(model, optimizer, images, make_pair)
            loss_history.append(loss)

            batch_size = images.shape[0]
            epoch_loss += loss * batch_size
            sample_count += batch_size

            if (step + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} "
                    f"Batch {step + 1}/{len(train_loader)} "
                    f"Mean loss: {epoch_loss / sample_count:.4f}"
                )
    return loss_history

