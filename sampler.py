import torch
import math

# generate data with Euler–Maruyama algorithm
# using ode
@torch.no_grad()
def sample_ode(model, noise, num_steps):
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

# using sde
@torch.no_grad()
def sample_sde(model, noise, num_steps, sigma_t):
    model.eval()
    x = noise.clone()
    t_min = 0.01
    t_max = 0.99
    h = (t_max - t_min) / num_steps
    for i in range(num_steps):
        t_value = t_min + i * h
        alpha = math.sin(math.pi / 2 * t_value)
        beta = math.cos(math.pi / 2 * t_value)
        sigma = sigma_t(t_value)
        b = math.pi / 2 * beta / alpha

        t = torch.full(
            (x.shape[0],), t_value,
            device=x.device, dtype=x.dtype
        )
        epsilon = torch.randn_like(x)
        x =  x + (b * x - (b + sigma * sigma / 2) * model(x, t) / beta) * h + sigma * math.sqrt(h) * epsilon
    return x