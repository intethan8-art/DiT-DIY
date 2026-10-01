import torch
import torch.nn as nn
import torch.nn.functional as F

# useful layers for transformer

# (B, N, C') -> (B, N, D) D: hidden dimension
class PatchEmbed(nn.Module):
    def __init__(self, C_patch, D) -> None:
        super().__init__()
        self.fc = nn.Linear(C_patch, D)

    def forward(self, x):
        return self.fc(x)

# position embedding using sine/cosine
class PositionEmbed(nn.Module):
    def __init__(self, grid_size, D) -> None:
        super().__init__()
        mode = D // 4
        grid_h, grid_w = grid_size 
        pos_emb = []
        k = torch.arange(mode, dtype=torch.float32)
        omega = 10000.0 ** (-k / mode)
        for r in range(grid_h):
            for c in range(grid_w):
                angle_r = omega * r
                angle_c = omega * c
                e_r_sin = torch.sin(angle_r)
                e_r_cos = torch.cos(angle_r)
                e_c_sin = torch.sin(angle_c)
                e_c_cos = torch.cos(angle_c)                   
                e_r_pairs = torch.stack([e_r_sin, e_r_cos], dim = -1)
                e_c_pairs = torch.stack([e_c_sin, e_c_cos], dim = -1)
                r_encoding = e_r_pairs.reshape(-1)
                c_encoding = e_c_pairs.reshape(-1)
                encoding = torch.cat([r_encoding, c_encoding], dim=-1)
                pos_emb.append(encoding)

        pos_emb = torch.stack(pos_emb, dim=0)
        self.register_buffer("pos_emb", pos_emb)

    def forward(self, x):
        return x + self.pos_emb