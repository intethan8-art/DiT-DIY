import torch

# generate data
@torch.no_grad()
def sample(model, noise, num_steps):
    model.eval()
    x = noise.clone()
    for i in range(num_steps):
        t = i / num_steps
        t = torch.full(
            (noise.shape[0],), i / num_steps,
            device=noise.device, dtype=noise.dtype
        )
        x =  x + model(x, t) / num_steps 
    return x