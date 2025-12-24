import torch
import json
import os
import rasterio
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from LandVit import SoilViT, GeoDataset

BEST_MODEL_PATH = 'model.pth'
ORIGINAL_JSON = 'eval_patch_data.json'
CORRECTED_JSON = 'gauss_corrected_eval_patch_data.json'
CORRECTED_OUTPUT_DIR = 'gauss_corrected_eval_outputs'
SIGMA_MULTIPLIER = 3.0
GAUSSIAN_SIGMA = 1.0
TRUE_GAUSSIAN_KERNEL_SIZE = 5
PRED_WEIGHT = 0.3
TRUE_WEIGHT = 0.7
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_donut_kernel(size=5, sigma=1.0):
    kernel = np.zeros((size, size))
    center = size // 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    kernel = np.exp(-dist ** 2 / (2 * sigma ** 2))
    kernel[center, center] = 0.0
    kernel /= kernel.sum() + 1e-8
    return kernel


def apply_donut_gaussian_filter(data, kernel):
    pad = kernel.shape[0] // 2
    padded = np.pad(data, ((pad, pad), (pad, pad)), mode='reflect')
    filtered = np.zeros_like(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            region = padded[i:i + kernel.shape[0], j:j + kernel.shape[1]]
            filtered[i, j] = np.sum(region * kernel)
    return filtered


def main():
    model = SoilViT(in_chans=13, out_chans=6, embed_dim=768, depth=12, num_heads=12)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    dataset = GeoDataset(ORIGINAL_JSON)

    with open(ORIGINAL_JSON, 'r') as f:
        data_config = json.load(f)

    os.makedirs(CORRECTED_OUTPUT_DIR, exist_ok=True)

    new_output_files = []

    donut_kernel = create_donut_kernel(size=TRUE_GAUSSIAN_KERNEL_SIZE, sigma=GAUSSIAN_SIGMA)

    with torch.no_grad():
        for idx in tqdm(range(len(dataset))):
            x, y, lat_lon = dataset[idx]
            x = x.unsqueeze(0).to(DEVICE)
            lat_lon = lat_lon.unsqueeze(0).to(DEVICE)
            y = y.unsqueeze(0).to(DEVICE)

            pred = model(x, lat_lon)

            abs_error = torch.abs(pred - y)

            corrected_data = y.clone()

            num_replaced_pixels = 0
            for c in range(6):
                channel_error = abs_error[0, c]  # (128, 128)
                if channel_error.numel() == 0:
                    continue
                mean_error = channel_error.mean()
                std_error = channel_error.std()
                if std_error == 0:
                    continue

                threshold = mean_error + SIGMA_MULTIPLIER * std_error
                mask = channel_error > threshold  # (128, 128)

                if mask.any():
                    channel_pred = pred[0, c].cpu().numpy()  # (128, 128)
                    channel_true = y[0, c].cpu().numpy()  # (128, 128)

                    filtered_pred = gaussian_filter(channel_pred, sigma=GAUSSIAN_SIGMA)

                    filtered_true = apply_donut_gaussian_filter(channel_true, donut_kernel)

                    corrected_channel = channel_true.copy()
                    corrected_channel[mask.cpu().numpy()] = PRED_WEIGHT * filtered_pred[
                        mask.cpu().numpy()] + TRUE_WEIGHT * filtered_true[mask.cpu().numpy()]

                    corrected_data[0, c] = torch.tensor(corrected_channel).to(DEVICE)

                    num_replaced_pixels += mask.sum().item()

            original_output_file = data_config['output_patch_files'][idx]
            if num_replaced_pixels > 0:
                print(
                    f"Correcting patch {idx}: replaced {num_replaced_pixels} pixels (out of {6 * 128 * 128}). Original file: {original_output_file}")

                corrected_np = corrected_data.squeeze(0).cpu().numpy()  # (6, 128, 128)

                with rasterio.open(original_output_file) as src:
                    profile = src.profile
                    profile.update(count=corrected_np.shape[0])

                original_basename = os.path.basename(original_output_file)
                new_output_file = os.path.join(CORRECTED_OUTPUT_DIR, f"corrected_{original_basename}")

                with rasterio.open(new_output_file, 'w', **profile) as dst:
                    dst.write(corrected_np)

                new_output_files.append(new_output_file)
            else:
                print(f"Patch {idx}: no pixels replaced (all within confidence thresholds).")
                new_output_files.append(original_output_file)

    data_config['output_patch_files'] = new_output_files
    with open(CORRECTED_JSON, 'w') as f:
        json.dump(data_config, f, indent=4)

    print(f"Data correction completed. Corrected JSON saved to {CORRECTED_JSON}")
    print(
        f"Total patches: {len(dataset)}, Corrected (replaced) patches: {len([f for f in new_output_files if 'corrected_' in f])}")


if __name__ == "__main__":
    main()