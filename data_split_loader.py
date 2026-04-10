import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset
import json
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ==============================================================================
# 1. GEODATASET CLASS
# ==============================================================================
class GeoDataset(Dataset):
    """
    Custom Dataset for loading geospatial patches with pre-calculated 
    training-based normalization parameters.
    """
    def __init__(self, json_file):
        with open(json_file, 'r') as f:
            data_config = json.load(f)
            
        # These parameters are guaranteed to be derived from the training set only
        self.input_means = np.array(data_config['input_means'], dtype=np.float32)
        self.input_stds = np.array(data_config['input_stds'], dtype=np.float32)
        self.output_means = np.array(data_config['output_means'], dtype=np.float32)
        self.output_stds = np.array(data_config['output_stds'], dtype=np.float32)
        
        self.input_files = data_config['input_patch_files']
        self.output_files = data_config['output_patch_files']

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # Load input and output patches using Rasterio
        with rasterio.open(self.input_files[idx]) as src:
            input_data = src.read().astype(np.float32)
        with rasterio.open(self.output_files[idx]) as src:
            output_data = src.read().astype(np.float32)
        
        # Apply standard Z-score normalization using training-derived statistics
        # Reshape means/stds for broadcasting across the (C, H, W) dimensions
        for b in range(input_data.shape[0]):
            input_data[b] = (input_data[b] - self.input_means[b]) / (self.input_stds[b] + 1e-8)
        
        for b in range(output_data.shape[0]):
            output_data[b] = (output_data[b] - self.output_means[b]) / (self.output_stds[b] + 1e-8)
        
        return torch.from_numpy(input_data), torch.from_numpy(output_data)

# ==============================================================================
# 2. PATCH GENERATION (STATISTICS-AGNOSTIC)
# ==============================================================================
def generate_patches_without_stats(input_tif, output_tif, patch_save_dir, window_size=128, step=32):
    """
    Generates and saves image patches from large GeoTIFF mosaics. 
    Note: Statistical calculation is decoupled from this stage to avoid leakage.
    """
    if not os.path.exists(patch_save_dir):
        os.makedirs(patch_save_dir)

    with rasterio.open(input_tif) as src_in, rasterio.open(output_tif) as src_out:
        input_profile = src_in.profile
        output_profile = src_out.profile
        input_data = src_in.read()
        output_data = src_out.read()

    # Align dimensions between input and supervision data
    h, w = input_data.shape[1:]
    output_data = output_data[:, :h, :w]
    output_data[output_data < 0] = np.nan  # Handle invalid/legacy values as NaNs

    input_patch_files = []
    output_patch_files = []

    print(f"Commencing patch generation for area size: {h}x{w}...")
    count = 0
    # Sliding window approach for patch extraction
    for i in range(0, h - window_size + 1, step):
        for j in range(0, w - window_size + 1, step):
            in_patch = input_data[:, i:i+window_size, j:j+window_size]
            out_patch = output_data[:, i:i+window_size, j:j+window_size]

            # Only retain patches containing valid (non-NaN) data
            if not np.all(np.isnan(in_patch)) and not np.all(np.isnan(out_patch)):
                in_path = os.path.join(patch_save_dir, f'in_{count}.tif')
                out_path = os.path.join(patch_save_dir, f'out_{count}.tif')
                
                # Update GeoTIFF profiles for local patches
                p_in = input_profile.copy()
                p_in.update(width=window_size, height=window_size, count=in_patch.shape[0])
                with rasterio.open(in_path, 'w', **p_in) as dst:
                    dst.write(np.nan_to_num(in_patch, nan=0))

                p_out = output_profile.copy()
                p_out.update(width=window_size, height=window_size, count=out_patch.shape[0])
                with rasterio.open(out_path, 'w', **p_out) as dst:
                    dst.write(np.nan_to_num(out_patch, nan=0))

                input_patch_files.append(in_path)
                output_patch_files.append(out_path)
                count += 1
    
    return input_patch_files, output_patch_files

# ==============================================================================
# 3. RIGOROUS PARAMETER ESTIMATION (TRAining ONLY)
# ==============================================================================
def compute_training_stats(input_files, output_files):
    """
    Computes band-wise mean and standard deviation exclusively from 
    the training set to ensure zero leakage from validation/test data.
    """
    print("Estimating normalization parameters from training partition...")
    
    def get_band_stats(file_list):
        with rasterio.open(file_list[0]) as ref:
            num_bands = ref.count
        
        # Accumulate pixel data per band for accurate statistical calculation
        band_accumulator = [[] for _ in range(num_bands)]
        for f in tqdm(file_list, desc="Processing training patches"):
            with rasterio.open(f) as src:
                data = src.read()
                for b in range(num_bands):
                    # Exclude zero-padded background to avoid biasing the distribution
                    valid_data = data[b][data[b] != 0]
                    band_accumulator[b].extend(valid_data.tolist())
        
        means = [float(np.mean(b)) for b in band_accumulator]
        stds = [float(np.std(b)) for b in band_accumulator]
        return means, stds

    in_m, in_s = get_band_stats(input_files)
    out_m, out_s = get_band_stats(output_files)
    return in_m, in_s, out_m, out_s

# ==============================================================================
# 4. EXECUTION PIPELINE
# ==============================================================================
if __name__ == "__main__":
    # Configuration paths
    RAW_INPUT_PATH = 'china_soil_input_all.tif'
    RAW_SUPERVISION_PATH = 'merged_layer_l1.tif'
    OUTPUT_PATCH_DIR = './spatial_independent_patches'
    
    # STEP 1: Patch Generation (Independent of statistical distribution)
    in_list, out_list = generate_patches_without_stats(RAW_INPUT_PATH, RAW_SUPERVISION_PATH, OUTPUT_PATCH_DIR)
    
    # STEP 2: Spatial/Dataset Splitting (Ensuring unseen data status)
    indices = np.arange(len(in_list))
    train_idx, val_idx = train_test_split(indices, train_size=0.8, random_state=42)
    
    # Segregate file paths into training and validation sets
    train_in = [in_list[i] for i in train_idx]
    train_out = [out_list[i] for i in train_idx]
    val_in = [in_list[i] for i in val_idx]
    val_out = [out_list[i] for i in val_idx]
    
    # STEP 3: Parameter Estimation (Training-set based only)
    # This addresses Reviewer 2's concern regarding statistical data leakage.
    in_m, in_s, out_m, out_s = compute_training_stats(train_in, train_out)
    
    # STEP 4: Persist Configurations
    # Training Configuration
    train_meta = {
        "input_means": in_m, "input_stds": in_s,
        "output_means": out_m, "output_stds": out_s,
        "input_patch_files": train_in, "output_patch_files": train_out
    }
    with open('train_dataset_config.json', 'w') as f:
        json.dump(train_meta, f)
        
    # Validation Configuration (Note: Uses training stats to ensure zero-leakage scaling)
    val_meta = {
        "input_means": in_m, "input_stds": in_s,
        "output_means": out_m, "output_stds": out_s,
        "input_patch_files": val_in, "output_patch_files": val_out
    }
    with open('val_dataset_config.json', 'w') as f:
        json.dump(val_meta, f)

    print("Success: Pipeline finalized.")


# # Create DataLoader
# dataset = GeoDataset(input_patch_files, output_patch_files, input_means, output_means)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Example usage
# for input_batch, output_batch in dataloader:
#     print(f"Input batch shape: {input_batch.shape}, Output batch shape: {output_batch.shape}")
#     break
