import torch
import torch.nn as nn
import torch.nn.functional as F
import layers
import patch_utils

# naive using MLP

class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x, t):
        inputs = torch.cat((x, t.reshape(-1, 1)), 1)
        x1 = F.relu(self.fc1(inputs))
        x2 = F.relu(self.fc2(x1))
        out = self.fc3(x2)
        return out
        
# build a dit model
# input: x (B, C, H, W), t (B)
# output: out(u_target) (B, C, H, W)
# x: Patchify - input_proj - PositionEmbed 
#                                           - TransformerBlock - output_proj - Unpatchify
# t:                         TimeEmbed

# original shape: B, C, H, W 
# patch-size: P 
# hidden-dimension: D 
# t_emb-hidden-dimension: t_D
# attention-head: h
# Block-hidden-dimension: b_D
# number of Blocks: Depth

class DIT(nn.Module):
    def __init__(self, C, H, W, P, D, t_D, h, b_D, Depth) -> None:
        super().__init__()

        self.patch_size = P 
        C_patch = C * P * P
        self.input_proj = nn.Linear(C_patch, D)
        self.output_proj = nn.Linear(D, C_patch)

        grid_size = (H // P, W // P)
        self.pos_emb = layers.PositionEmbed(grid_size, D)
        self.t_emb = layers.TimeEmbed(D, t_D)
        self.blocks = nn.ModuleList(
            [layers.TransformerBlock(D, h, b_D)
            for _ in range(Depth)]
        )
        
        self.y_emb = nn.Embedding(11, D)
        self.null_label = 10

# classifier-free guidance
    def forward(self, x, t, y):
        original_size = x.shape
        x_patchified = patch_utils.patchify(x, self.patch_size)
        x_emb = self.pos_emb(self.input_proj(x_patchified))
        t_emb = self.t_emb(t)
        y_emb = self.y_emb(y)
        out1 = x_emb
        for block in self.blocks:
            out1 = block(out1, t_emb, y_emb)
        out2 = self.output_proj(out1)
        out = patch_utils.unpatchify(out2, self.patch_size, original_size)
        return out