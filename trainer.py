import torch
from flow import make_training_pair
import torch.nn.functional as F
from data import sample_data

# training process for one batch
def train_step(model, optimizer, z):
    optimizer.zero_grad()
    x_t, t, u_target = make_training_pair(z)
    u_pred = model(x_t, t)
    loss = F.mse_loss(u_pred, u_target)
    loss.backward()
    optimizer.step()
    return loss.item()

def train_loops(model, optimizer, num_steps, batch_size):
    loss_history = []
    for _ in range(num_steps):
        z = sample_data(batch_size) 
        loss = train_step(model, optimizer, z)
        loss_history.append(loss)
    return loss_history
        