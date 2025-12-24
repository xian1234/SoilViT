import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.vision_transformer import VisionTransformer
from torch.utils.data import Dataset
import numpy as np
import rasterio
import json
from math import exp

class GeoPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_value=90.0):
        super().__init__()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.register_buffer('div_term', div_term)
        self.d_model = d_model
        self.max_value = max_value

    def forward(self, lat_lon):
        B = lat_lon.shape[0]
        lat_lon = lat_lon / self.max_value
        pos_enc = torch.zeros(B, 4, self.d_model, device=lat_lon.device)
        for i in range(4):
            pos_enc[:, i, 0::2] = torch.sin(lat_lon[:, i].unsqueeze(1) * self.div_term)
            pos_enc[:, i, 1::2] = torch.cos(lat_lon[:, i].unsqueeze(1) * self.div_term)
        pos_enc = pos_enc.mean(dim=1, keepdim=True)
        return pos_enc

class ResidualDecoderBlock(nn.Module):
    """
    Replace your pure transposed conv decoder with residual upsampling + attention to preserve details
    and prevent over-smoothing.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

    def forward(self, x):
        x = self.upsample(x)
        residual = x
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x) + residual)
        return x

# Attention-based skip connection block
class AttentionSkipBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.attention = nn.Sequential(
            nn.Conv2d(out_channels, out_channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 8, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv(x)
        attn = self.attention(out)
        return out * attn + out  # Residual attention

# Spectral consistency loss: encourages consistent band ratios across outputs
class SpectralConsistencyLoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction):
        # prediction: (B, 7, H, W)
        B, C, H, W = prediction.shape
        ratios = prediction[:, :, :, :].unsqueeze(2) / (prediction.sum(dim=1, keepdim=True).unsqueeze(2) + self.epsilon)
        spectral_diff = torch.std(ratios, dim=1).mean()
        return spectral_diff

class SSIMLoss(nn.Module):
    """结构相似性损失"""
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self._create_window(window_size, self.channel)

    def _create_window(self, window_size, channel):
        # 创建高斯窗口
        sigma = 1.5
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        (_, channel, height, width) = img1.size()
        
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = self._create_window(self.window_size, channel)
            window = window.to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=self.window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=self.window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=self.window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if self.size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

class R2Loss(nn.Module):
    """R2分数损失"""
    def __init__(self):
        super(R2Loss, self).__init__()
        
    def forward(self, y_pred, y_true):
        SS_res = torch.sum(torch.square(y_true - y_pred), dim=[1,2,3])
        SS_tot = torch.sum(torch.square(y_true - torch.mean(y_true, dim=[1,2,3], keepdim=True)), dim=[1,2,3])
        return 1 - SS_res / (SS_tot + 1e-8)

class CombinedLoss(nn.Module):
    """组合损失函数"""
    def __init__(self, mse_weight=0.5, ssim_weight=0.3, r2_weight=0.2, channel_weights=None):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.ssim = SSIMLoss()
        self.r2 = R2Loss()
        
        # 损失权重
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.r2_weight = r2_weight
        
        # 通道权重
        self.channel_weights = channel_weights
        if channel_weights is not None:
            self.channel_weights = torch.tensor(channel_weights, dtype=torch.float32)
        
    def forward(self, y_pred, y_true):
        # 计算各损失
        mse_loss = self.mse(y_pred, y_true)
        
        # SSIM损失需要最大化，所以取1-SSIM
        ssim_loss = 1 - self.ssim(y_pred, y_true)
        
        # R2损失需要最小化，所以直接使用
        r2_loss = self.r2(y_pred, y_true).mean()
        
        # 应用通道权重（如果有）
        if self.channel_weights is not None:
            self.channel_weights = self.channel_weights.to(y_pred.device)
            # 假设通道维度是1
            channel_dim = 1
            mse_loss = (mse_loss * self.channel_weights).sum() / self.channel_weights.sum()
        
        # 组合损失
        combined_loss = (self.mse_weight * mse_loss + 
                         self.ssim_weight * ssim_loss + 
                         self.r2_weight * r2_loss)
        
        return combined_loss


# Updated decoder with attention skip blocks
class AttentionDecoder(nn.Module):
    def __init__(self, embed_dim, out_chans):
        super().__init__()
        self.decode_layers = nn.Sequential(
            AttentionSkipBlock(embed_dim, 256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            AttentionSkipBlock(256, 128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            AttentionSkipBlock(128, 64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            AttentionSkipBlock(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, out_chans, kernel_size=3, padding=1),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.decode_layers(x)


# Vision Transformer with Geographic Encoding
class SoilViT(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_chans=14, out_chans=7, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.geo_pos_enc = GeoPositionalEncoding(embed_dim)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=0.1, activation='gelu', norm_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # Improved Decoder:
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
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)

    def forward(self, x, lat_lon):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        geo_enc = self.geo_pos_enc(lat_lon).expand(-1, self.num_patches + 1, -1)
        x = x + geo_enc
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        x = x[:, 1:]
        x = rearrange(x, 'b (h w) c -> b c h w', h=self.img_size // self.patch_size)
        x = self.decoder(x)
        return x


# Custom Dataset with Latitude/Longitude
class GeoDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data_config = json.load(f)
        self.input_means = data_config['input_means']
        self.input_stds = data_config['input_stds']
        self.output_means = data_config['output_means']
        self.output_stds = data_config['output_stds']
        self.input_files = data_config['input_patch_files']
        self.output_files = data_config['output_patch_files']

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load input image (14 bands)
        with rasterio.open(self.input_files[idx]) as src:
            input_data = src.read()  # Shape: (14, 128, 128)
            transform = src.transform
            left_top = transform * (0, 0)  # Top-left corner
            right_bottom = transform * (src.width, src.height)  # Bottom-right corner
        lat_lon = np.array([left_top[1], left_top[0], right_bottom[1], right_bottom[0]])  # [lat1, lon1, lat2, lon2]

        # Load output image (7 bands)
        with rasterio.open(self.output_files[idx]) as src:
            output_data = src.read()  # Shape: (7, 128, 128)
        
        # normalize input and output data
        for band in range(input_data.shape[0]):
            input_data[band] = (input_data[band] - self.input_means[band]) / (self.input_stds[band] + 1e-8)
        
        # for band in range(output_data.shape[0]):
        #     output_data[band] = (output_data[band] - self.output_means[band]) / (self.output_stds[band] + 1e-8)

        # Convert to tensors
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        if output_data.shape[0] == 7:
            output_tensor = torch.tensor(output_data[[0,1,3,4,5,6],:,:], dtype=torch.float32)
        else:
            output_tensor = torch.tensor(output_data[[0,1,2,3,4,5],:,:], dtype=torch.float32)
        lat_lon_tensor = torch.tensor(lat_lon, dtype=torch.float32)

        return input_tensor, output_tensor, lat_lon_tensor



# Example Usage
if __name__ == "__main__":
    # Model configurations
    base_model = SoilViT(embed_dim=768, depth=12, num_heads=12)  # Transformer-Base
    # large_model = GeoViT(embed_dim=1024, depth=24, num_heads=16)  # Transformer-Large

    # Dummy input
    input_img = torch.randn(1, 14, 128, 128)
    lat_lon = torch.randn(1, 4)

    # Forward pass
    output = base_model(input_img, lat_lon)
    print(f"Output shape: {output.shape}")  # Expected: (1, 7, 128, 128)

    # Verify sum-to-1 constraint
    band_sum = output.sum(dim=1)  # Sum across 7 bands
    print(f"Sum of bands per pixel (should be close to 1): {band_sum[0, 0, 0]}")

    # Loss function
    criterion = nn.KLDivLoss(reduction='batchmean')
    target = torch.randn(1, 7, 128, 128).softmax(dim=1)  # Dummy target
    loss = criterion(output, target)
    print(f"Loss: {loss.item()}") #load_shapefile_with_index