import torch
import torch.nn as nn
import torch.nn.functional as F
import math
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

# multihead self-attention 
class SelfAttention(nn.Module):
    def __init__(self, D, h) -> None:
        super().__init__()
        self.h = h
        self.d = D // h 
        self.W_q = nn.Linear(D, D)
        self.W_k = nn.Linear(D, D)
        self.W_v = nn.Linear(D, D)
        self.W_o = nn.Linear(D, D)

    def forward(self, x):
        B, N, D = x.shape
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        q = q.reshape(B, N, self.h, self.d)
        q = q.permute(0, 2, 1, 3)
        k = k.reshape(B, N, self.h, self.d)
        k = k.permute(0, 2, 3, 1)
        weights = torch.softmax(q @ k / math.sqrt(self.d), -1)
        v = v.reshape(B, N, self.h, self.d)
        v = v.permute(0, 2, 1, 3)
        out = weights @ v
        out = out.permute(0, 2, 1, 3)
        out = out.reshape(B, N, D)
        out = self.W_o(out)
        return out