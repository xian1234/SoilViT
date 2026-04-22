"""
SoilViT model definition, dataset, and loss utilities.

Previously hard-coded constants (dropout, geo_max_value, output band indices)
are now explicit constructor arguments, all with documented defaults that
match the published configuration in config/defaults.yaml.
"""

import json
from math import exp

import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset


# ---------------------------------------------------------------------------
# Geographic positional encoding
# ---------------------------------------------------------------------------

class GeoPositionalEncoding(nn.Module):
    """Sinusoidal encoding of (lat1, lon1, lat2, lon2) patch corners."""

    def __init__(self, d_model: int, max_value: float = 90.0):
        super().__init__()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        self.register_buffer("div_term", div_term)
        self.d_model   = d_model
        self.max_value = max_value

    def forward(self, lat_lon: torch.Tensor) -> torch.Tensor:
        B = lat_lon.shape[0]
        lat_lon = lat_lon / self.max_value
        pos_enc = torch.zeros(B, 4, self.d_model, device=lat_lon.device)
        for i in range(4):
            pos_enc[:, i, 0::2] = torch.sin(lat_lon[:, i].unsqueeze(1) * self.div_term)
            pos_enc[:, i, 1::2] = torch.cos(lat_lon[:, i].unsqueeze(1) * self.div_term)
        return pos_enc.mean(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Decoder blocks
# ---------------------------------------------------------------------------

class AttentionSkipBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, max(1, out_channels // 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, out_channels // 8), out_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out  = self.conv(x)
        attn = self.attention(out)
        return out * attn + out


class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim: int, out_chans: int):
        super().__init__()
        self.decode_layers = nn.Sequential(
            AttentionSkipBlock(embed_dim, 256),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            AttentionSkipBlock(256, 128),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            AttentionSkipBlock(128, 64),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            AttentionSkipBlock(64, 32),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(32, out_chans, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode_layers(x)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class SoilViT(nn.Module):
    """
    Vision Transformer for soil property mapping.

    Parameters
    ----------
    img_size      : spatial size of each input patch (pixels), default 128
    patch_size    : ViT tokenisation patch size, default 16
    in_chans      : number of input spectral bands, default 13
    out_chans     : number of predicted soil properties, default 6
    embed_dim     : transformer embedding dimension, default 768
    depth         : number of transformer encoder layers, default 12
    num_heads     : number of self-attention heads, default 12
    dropout       : dropout rate in transformer blocks, default 0.1
    geo_max_value : normalisation divisor for lat/lon coordinates, default 90.0
    """

    def __init__(
        self,
        img_size: int   = 128,
        patch_size: int = 16,
        in_chans: int   = 13,
        out_chans: int  = 6,
        embed_dim: int  = 768,
        depth: int      = 12,
        num_heads: int  = 12,
        dropout: float  = 0.1,
        geo_max_value: float = 90.0,
    ):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim,
                                     kernel_size=patch_size, stride=patch_size)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop   = nn.Dropout(dropout)
        self.geo_pos_enc = GeoPositionalEncoding(embed_dim, max_value=geo_max_value)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation="gelu",
                norm_first=True,
            )
            for _ in range(depth)
        ])
        self.norm    = nn.LayerNorm(embed_dim)
        self.decoder = AttentionDecoder(embed_dim=embed_dim, out_chans=out_chans)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor, lat_lon: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat((cls, x), dim=1) + self.pos_embed
        x   = x + self.geo_pos_enc(lat_lon).expand(-1, self.num_patches + 1, -1)
        x   = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)[:, 1:]
        x = rearrange(x, "b (h w) c -> b c h w",
                      h=self.img_size // self.patch_size)
        return self.decoder(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GeoDataset(Dataset):
    """
    Loads patch pairs from a JSON config produced by spatial_data_split.py.

    Parameters
    ----------
    json_file           : path to the split config JSON
    output_band_indices : which bands to select from the label raster.
                          Defaults to [0,1,3,4,5,6] (excludes derived band 2).
                          Set via config/defaults.yaml → data.output_band_indices.
    """

    def __init__(self, json_file: str, output_band_indices: list = None):
        with open(json_file) as f:
            cfg = json.load(f)
        self.input_means  = cfg["input_means"]
        self.input_stds   = cfg["input_stds"]
        self.input_files  = cfg["input_patch_files"]
        self.output_files = cfg["output_patch_files"]
        self.band_indices = output_band_indices if output_band_indices is not None \
                            else [0, 1, 3, 4, 5, 6]

    def __len__(self) -> int:
        return len(self.input_files)

    def __getitem__(self, idx: int):
        with rasterio.open(self.input_files[idx]) as src:
            input_data = src.read().astype(np.float32)
            t  = src.transform
            tl = t * (0, 0)
            br = t * (src.width, src.height)

        lat_lon = np.array([tl[1], tl[0], br[1], br[0]], dtype=np.float32)

        with rasterio.open(self.output_files[idx]) as src:
            output_data = src.read().astype(np.float32)

        for b in range(input_data.shape[0]):
            input_data[b] = (input_data[b] - self.input_means[b]) \
                            / (self.input_stds[b] + 1e-8)

        return (
            torch.from_numpy(input_data),
            torch.from_numpy(output_data[self.band_indices]),
            torch.from_numpy(lat_lon),
        )


# ---------------------------------------------------------------------------
# Auxiliary losses (kept for optional use)
# ---------------------------------------------------------------------------

class SpectralConsistencyLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor) -> torch.Tensor:
        total = prediction.sum(dim=1, keepdim=True) + self.epsilon
        ratios = prediction.unsqueeze(2) / total.unsqueeze(2)
        return torch.std(ratios, dim=1).mean()


class SSIMLoss(nn.Module):
    def __init__(self, window_size: int = 11, size_average: bool = True):
        super().__init__()
        self.window_size  = window_size
        self.size_average = size_average
        self.channel = 1
        self.window  = self._make_window(window_size, 1)

    def _make_window(self, size: int, channel: int) -> torch.Tensor:
        sigma = 1.5
        g = torch.tensor([exp(-(x - size // 2) ** 2 / (2 * sigma ** 2))
                           for x in range(size)])
        w = (g.unsqueeze(1) @ g.unsqueeze(0)).expand(channel, 1, size, size).contiguous()
        return w

    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        _, ch, _, _ = img1.size()
        if ch != self.channel or self.window.dtype != img1.dtype:
            self.window  = self._make_window(self.window_size, ch).to(img1.device, img1.dtype)
            self.channel = ch
        pad = self.window_size // 2
        mu1 = F.conv2d(img1, self.window, padding=pad, groups=ch)
        mu2 = F.conv2d(img2, self.window, padding=pad, groups=ch)
        s1  = F.conv2d(img1 * img1, self.window, padding=pad, groups=ch) - mu1 ** 2
        s2  = F.conv2d(img2 * img2, self.window, padding=pad, groups=ch) - mu2 ** 2
        s12 = F.conv2d(img1 * img2, self.window, padding=pad, groups=ch) - mu1 * mu2
        C1, C2 = 0.01 ** 2, 0.03 ** 2
        ssim = ((2 * mu1 * mu2 + C1) * (2 * s12 + C2)) \
             / ((mu1 ** 2 + mu2 ** 2 + C1) * (s1 + s2 + C2))
        return ssim.mean() if self.size_average else ssim.mean(1).mean(1).mean(1)


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight: float = 0.5, ssim_weight: float = 0.3,
                 r2_weight: float = 0.2, channel_weights: list = None):
        super().__init__()
        self.mse  = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.mse_weight  = mse_weight
        self.ssim_weight = ssim_weight
        self.r2_weight   = r2_weight
        self.channel_weights = (torch.tensor(channel_weights, dtype=torch.float32)
                                if channel_weights else None)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        mse  = self.mse(y_pred, y_true)
        ssim = 1 - self.ssim(y_pred, y_true)
        ss_res = torch.sum((y_true - y_pred) ** 2,    dim=[1, 2, 3])
        ss_tot = torch.sum((y_true - y_true.mean(dim=[1, 2, 3], keepdim=True)) ** 2,
                           dim=[1, 2, 3])
        r2 = (1 - ss_res / (ss_tot + 1e-8)).mean()
        return self.mse_weight * mse + self.ssim_weight * ssim + self.r2_weight * r2
