import torch

# construct the probability path

def make_training_pair(z):
    eps = torch.randn_like(z)
    batch_size = z.shape[0]
    t = torch.rand(batch_size, device=z.device, dtype=z.dtype)
    broadcast_shape = (batch_size,) + (1,) * (z.ndim - 1)
    t_broadcast = t.reshape(broadcast_shape)
    x_t = t_broadcast * z + (1 - t_broadcast) * eps
    u_target = z - eps
    return x_t, t, u_target

