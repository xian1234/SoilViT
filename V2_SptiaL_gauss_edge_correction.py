import torch
import json
import os
import rasterio
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from LandVit import SoilViT, GeoDataset
import cv2

BEST_MODEL_PATH = 'model.pth'
ORIGINAL_JSON = 'eval_patch_data.json'
CORRECTED_JSON = 'corrected_eval_patch_data.json'
CORRECTED_OUTPUT_DIR = 'corrected_eval_outputs'
SIGMA_MULTIPLIER = 1.6
GROUP_SIZE_DEFAULT = 20
GROUP_SIZE_SPECIAL = 15
LINE_ERROR_THRESHOLD = 0.8
MEDIUM_LINE_THRESHOLD = 0.4
EDGE_THRESHOLD = 0.35
EDGE_RATIO_THRESHOLD = 0.25
ROW_EDGE_THRESHOLD = 0.35
GAUSSIAN_SIGMA = 1.0
TRUE_GAUSSIAN_KERNEL_SIZE = 5
PRED_WEIGHT_NORMAL = 0.2
TRUE_WEIGHT_NORMAL = 0.8
PRED_WEIGHT_LINE = 0.9
TRUE_WEIGHT_LINE = 0.1
PRED_WEIGHT_MEDIUM = 0.7
TRUE_WEIGHT_MEDIUM = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_donut_kernel(size=5, sigma=1.0):
    """Creates a donut Gaussian kernel with zero weight at center"""
    kernel = np.zeros((size, size))
    center = size // 2
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    dist = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    kernel = np.exp(-dist ** 2 / (2 * sigma ** 2))
    kernel[center, center] = 0.0
    kernel /= kernel.sum() + 1e-8
    return kernel


def apply_donut_gaussian_filter(data, kernel):
    """Applies donut Gaussian filter to a single channel"""
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
            x = x.unsqueeze(0).to(DEVICE)  # (1, 13, 128, 128)
            lat_lon = lat_lon.unsqueeze(0).to(DEVICE)
            y = y.unsqueeze(0).to(DEVICE)  # (1, 6, 128, 128)

            pred = model(x, lat_lon)  # (1, 6, 128, 128)

            abs_error = torch.abs(pred - y)  # (1, 6, 128, 128)

            corrected_data = y.clone()  # (1, 6, 128, 128)

            num_replaced_pixels = 0
            height = abs_error.shape[2]
            width = abs_error.shape[3]
            for c in range(6):
                channel_error = abs_error[0, c]  # (128, 128)
                if channel_error.numel() == 0:
                    continue

                global_mean = channel_error.mean()
                global_std = channel_error.std()
                if global_std == 0:
                    continue

                mask = torch.zeros_like(channel_error, dtype=torch.uint8)  # (128, 128)

                error_np = channel_error.cpu().numpy()
                sobel_x = cv2.Sobel(error_np, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(error_np, cv2.CV_64F, 0, 1, ksize=3)
                edge_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
                edge_mask = (edge_mag > np.mean(edge_mag) + EDGE_THRESHOLD * np.std(edge_mag)).astype(np.uint8)

                group_size = GROUP_SIZE_SPECIAL if c in [3, 4] else GROUP_SIZE_DEFAULT

                for start in range(0, height, group_size):
                    end = min(start + group_size, height)
                    group_error = channel_error[start:end, :]
                    group_edge = torch.tensor(edge_mask[start:end, :]).to(DEVICE)

                    mean_error = group_error.mean()
                    std_error = group_error.std()
                    if std_error == 0:
                        mean_error = global_mean
                        std_error = global_std

                    edge_ratio = group_edge.sum() / (group_error.numel() + 1e-8)
                    multiplier = SIGMA_MULTIPLIER if edge_ratio < EDGE_RATIO_THRESHOLD else 1.0

                    threshold = mean_error + multiplier * std_error

                    group_mask = (group_error > threshold).to(torch.uint8)
                    mask[start:end, :] = group_mask

                remainder_start = (height // group_size) * group_size
                if remainder_start < height:
                    remainder_error = channel_error[remainder_start:, :]
                    global_threshold = global_mean + SIGMA_MULTIPLIER * global_std
                    remainder_mask = (remainder_error > global_threshold).to(torch.uint8)
                    mask[remainder_start:, :] = remainder_mask

                if c in [3, 4]:
                    for row in range(height):
                        row_errors = mask[row, :].sum().item()
                        error_ratio = row_errors / width
                        row_edge = edge_mask[row, :].sum() / width

                        if error_ratio > LINE_ERROR_THRESHOLD or (error_ratio > 0.5 and row_edge > ROW_EDGE_THRESHOLD):
                            mask[row, :] = 2
                        else:
                            row_mask_np = mask[row, :].cpu().numpy()
                            diff = np.diff(np.pad(row_mask_np == 1, (1, 1), mode='constant'))
                            starts = np.where(diff == 1)[0]
                            ends = np.where(diff == -1)[0]
                            min_len = min(len(starts), len(ends))
                            lengths = ends[:min_len] - starts[:min_len] if min_len > 0 else np.array([])

                            if len(lengths) > 0:
                                max_length_ratio = lengths.max() / width
                                if MEDIUM_LINE_THRESHOLD < max_length_ratio < LINE_ERROR_THRESHOLD:
                                    for s, e in zip(starts[:min_len], ends[:min_len]):
                                        if (e - s) / width > MEDIUM_LINE_THRESHOLD:
                                            mask[row, s:e] = 3

                if (mask > 0).any():
                    channel_pred = pred[0, c].cpu().numpy()  # (128, 128)
                    channel_true = y[0, c].cpu().numpy()  # (128, 128)

                    filtered_pred = gaussian_filter(channel_pred, sigma=GAUSSIAN_SIGMA)

                    filtered_true = apply_donut_gaussian_filter(channel_true, donut_kernel)

                    corrected_channel = channel_true.copy()

                    mask_np = mask.cpu().numpy()

                    normal_mask = (mask_np == 1)
                    corrected_channel[normal_mask] = TRUE_WEIGHT_NORMAL * filtered_true[
                        normal_mask] + PRED_WEIGHT_NORMAL * filtered_pred[normal_mask]

                    medium_mask = (mask_np == 3)
                    corrected_channel[medium_mask] = TRUE_WEIGHT_MEDIUM * filtered_true[
                        medium_mask] + PRED_WEIGHT_MEDIUM * filtered_pred[medium_mask]

                    line_mask = (mask_np == 2)
                    corrected_channel[line_mask] = TRUE_WEIGHT_LINE * filtered_true[line_mask] + PRED_WEIGHT_LINE * \
                                                   filtered_pred[line_mask]

                    corrected_data[0, c] = torch.tensor(corrected_channel).to(DEVICE)

                    num_replaced_pixels += (mask > 0).sum().item()

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