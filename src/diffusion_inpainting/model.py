"""Compact class-conditional U-Net for 16x16 grayscale images."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freq = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / max(half - 1, 1))
    angles = t.float()[:, None] * freq[None]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    return F.pad(embedding, (0, dim - embedding.shape[-1]))


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, in_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.emb = nn.Linear(emb_dim, out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.emb(F.silu(emb))[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class ConditionalUNet(nn.Module):
    def __init__(self, channels: int = 32, emb_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.emb_dim = emb_dim
        self.time_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.class_emb = nn.Embedding(num_classes, emb_dim)
        self.input = nn.Conv2d(1, channels, 3, padding=1)
        self.down_block = ResBlock(channels, channels, emb_dim)
        self.downsample = nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1)
        self.mid = ResBlock(channels * 2, channels * 2, emb_dim)
        self.upsample = nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1)
        self.up_block = ResBlock(channels * 2, channels, emb_dim)
        self.out_norm = nn.GroupNorm(8, channels)
        self.output = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        emb = self.time_mlp(timestep_embedding(t, self.emb_dim)) + self.class_emb(labels)
        skip = self.down_block(self.input(x), emb)
        h = self.mid(self.downsample(skip), emb)
        h = self.up_block(torch.cat([self.upsample(h), skip], dim=1), emb)
        return self.output(F.silu(self.out_norm(h)))
