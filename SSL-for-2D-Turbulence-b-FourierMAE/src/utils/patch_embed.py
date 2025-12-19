# PatchEmbed code from IBM/NASA's Prithvi

import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import to_2tuple


class PatchEmbed(nn.Module):
  """ Patch embedding: (B,C,T,H,W) -> (B,C,L) -> (B,L,C)
  B: batch
  C: channels
  T: num frames (in time)
  H,W: frame height width
  """

  def __init__(
      self,
      img_size=224,
      patch_size=16,
      num_frames=1,
      tubelet_size=1,
      in_chans=1,
      embed_dim=384,
      norm_layer=None,
      flatten=True,
      bias=True,
  ):
      super().__init__()
      img_size = to_2tuple(img_size)
      patch_size = to_2tuple(patch_size)
      self.img_size = img_size
      self.patch_size = patch_size
      self.num_frames = num_frames
      self.tubelet_size = tubelet_size
      self.grid_size = (num_frames // tubelet_size, img_size[0] // patch_size[0], img_size[1] // patch_size[1])
      self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
      self.flatten = flatten

      self.proj = nn.Conv3d(in_chans, embed_dim,
                            kernel_size=(tubelet_size, patch_size[0], patch_size[1]),
                            stride=(tubelet_size, patch_size[0], patch_size[1]),
                            bias=bias)
      self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

  def forward(self, x):
    B, C, T, H, W = x.shape
    x = self.proj(x)
    if self.flatten:
        x = x.flatten(2).transpose(1, 2)    # B,C,T,H,W -> B,L,C
    x = self.norm(x)
    return x