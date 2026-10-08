import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# useful layers for transformer

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

class TimeEmbed(nn.Module):
    def __init__(self, D, H) -> None:
        super().__init__()
        mode = D // 2
        k = torch.arange(mode, dtype=torch.float32)
        omega = 10000.0 ** (-k / mode)
        self.register_buffer("omega", omega)

        # a learnable MLP
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, D)

    def forward(self, t):
        angles = t[:, None] * self.omega[None, :]
        t_sin = torch.sin(angles)
        t_cos = torch.cos(angles)
        pairs = torch.stack([t_sin, t_cos], dim=-1)
        t_emb = pairs.flatten(start_dim=1)
        t_emb = self.fc2(F.gelu(self.fc1(t_emb)))
        return t_emb

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

# x' = x + attention(LN1(x)) 
# x" = x'+ MLP(LN2(x)) 
# H1: hidden dimension h: head H2: hidden dimension for t

# 再考虑加上时间调制 
# 1) layernorm 的 scale 和 shift 由 t 来调制
# 2) 加入门控
class TransformerBlock(nn.Module):
    def __init__(self, D, h, H) -> None:
        super().__init__()
        self.attn = SelfAttention(D, h)
        self.fc1 = nn.Linear(D, H)
        self.fc2 = nn.Linear(H, D)
        self.norm1 = nn.LayerNorm(D, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(D, elementwise_affine=False)
        self.proj_t = nn.Linear(D, 6*D)
        nn.init.zeros_(self.proj_t.weight)
        nn.init.zeros_(self.proj_t.bias)

    def forward(self, x, t_emb, y_emb):
        condition = t_emb + y_emb
        t_mod = self.proj_t(condition)
        scale_attn, shift_attn, gate_attn, scale_mlp, shift_mlp, gate_mlp = t_mod.chunk(6, dim=-1)
        x_modulated = (
            self.norm1(x) * (1 + scale_attn[:, None, :])
            + shift_attn[:, None, :]
        )
        x1 = x + gate_attn[:, None, :] * self.attn(x_modulated)
        x1_modulated = (
            self.norm2(x1) * (1 + scale_mlp[:, None, :])
            + shift_mlp[:, None, :]
        )
        out = x1 + gate_mlp[:, None, :] * self.fc2(F.gelu(self.fc1(x1_modulated)))
        return out
    