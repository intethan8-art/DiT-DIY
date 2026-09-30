# (B, C, H, W) -> (B, N, C') 
# N = (H / P) * (W / P)
# C' = C * P * P 
def patchify(x_batch, P):
    B, C, H, W = x_batch.shape
    x = x_batch.reshape(B, C, H // P, P, W // P, P)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, (H // P) * (W // P), C * P * P)
    return x


# image_size = (B, C, H, W)
def unpatchify(patches, P, image_size):
    B, C, H, W = image_size
    out = patches.reshape(B, H // P, W // P, C, P, P)
    out = out.permute(0, 3, 1, 4, 2, 5)
    out = out.reshape(B, C, H, W)
    return out