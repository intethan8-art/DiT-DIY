import torch
import torch.nn as nn
import layers
import patch_utils

# build a dit model
# input: x, t
# output: out(u_target)
# x: Patchify - PatchEmbed - PositionEmbed 
#                                              - TransformerBlock - ?(MLP projection) - Unpatchify
# t:                         TimeEmbed
class DIT(nn.Module):
    def __init__() -> None:
        super().__init__()

    def forward():
        return