import rasterio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
import os


# Define a Dataset class for DataLoader
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
        with rasterio.open(self.input_files[idx]) as src:
            input_data = src.read()
        with rasterio.open(self.output_files[idx]) as src:
            output_data = src.read()
        
        for band in range(input_data.shape[0]):
            input_data[band] = (input_data[band] - self.input_means[band]) / (self.input_stds[band] + 1e-8)
        
        for band in range(output_data.shape[0]):
            output_data[band] = (output_data[band] - self.output_means[band]) / (self.output_stds[band] + 1e-8)
        
        input_tensor = torch.from_numpy(input_data).float()
        output_tensor = torch.from_numpy(output_data).float()
        return input_tensor, output_tensor


def generate_all_data():
    # Step 1: Open and align the input and output GeoTIFF files
    input_file = '/nfs-data3/zlx/land/LandAI/china_soil_input_all.tif'  # Model input data
    output_file = '/nfs-data3/zlx/land/LandAI/output/merged_layers/merged_layer_l1.tif'  # Model supervision data
    output_dir = '/nfs-data3/zlx/land/LandAI/used_patches'  # Directory to save patches
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with rasterio.open(input_file) as input_src:
        input_data = input_src.read()
        input_shape = input_data.shape[1:]  # (height, width)
        input_profile = input_src.profile

    with rasterio.open(output_file) as output_src:
        output_data = output_src.read()
        output_shape = output_data.shape[1:]  # (height, width)
        output_profile = output_src.profile

    # Align output data size to input data size
    if input_shape != output_shape:
        min_height = min(input_shape[0], output_shape[0])
        min_width = min(input_shape[1], output_shape[1])
        output_data = output_data[:, :min_height, :min_width]
        if output_shape[0] < input_shape[0] or output_shape[1] < input_shape[1]:
            # Pad with zeros if output is smaller than input
            pad_height = input_shape[0] - min_height
            pad_width = input_shape[1] - min_width
            output_data = np.pad(output_data, ((0, 0), (0, pad_height), (0, pad_width)), mode='constant', constant_values=0)

    # Step 2: Calculate mean for each band (for normalization)
    output_data[output_data < 0] = np.nan  # Replace -9999 with NaN for valid data calculation
    input_means = [float(np.nanmean(input_data[band])) for band in range(input_data.shape[0])]
    input_stds = [float(np.nanstd(input_data[band])) for band in range(input_data.shape[0])]
    output_means = [float(np.nanmean(output_data[band])) for band in range(output_data.shape[0])]
    output_stds = [float(np.nanstd(output_data[band])) for band in range(output_data.shape[0])]

    print(f"Input data mean: {input_means}, Output data mean: {output_means}")

    # Step 3: Crop data into 128x128 patches with 32x32 overlap
    window_size = 128
    step = 32
    input_patches = []
    output_patches = []

    height, width = input_shape
    for i in range(0, height, step):
        for j in range(0, width, step):
            # Adjust the starting point for the last patches to ensure 128x128 size
            start_i = min(i, height - window_size) if i + window_size > height else i
            start_j = min(j, width - window_size) if j + window_size > width else j
            input_patch = input_data[:, start_i:start_i + window_size, start_j:start_j + window_size]
            output_patch = output_data[:, start_i:start_i + window_size, start_j:start_j + window_size]
            
            # Step 4: Only keep patches with valid data (not all NaN), replace NaN with 0
            if not np.all(np.isnan(input_patch)) and not np.all(np.isnan(output_patch)):
                input_patch = np.nan_to_num(input_patch, nan=0)
                output_patch = np.nan_to_num(output_patch, nan=0)
                input_patches.append(input_patch)
                output_patches.append(output_patch)

    # Save patches as GeoTIFF files with geolocation information
    input_patch_files = []
    output_patch_files = []
    for idx, (input_patch, output_patch) in enumerate(zip(input_patches, output_patches)):
        input_patch_file = os.path.join(output_dir, f'input_patch_{idx}.tif')
        output_patch_file = os.path.join(output_dir, f'output_patch_{idx}.tif')
        
        # Update profile for patches
        profile = input_profile.copy()
        profile.update(width=window_size, height=window_size, count=input_patch.shape[0])
        with rasterio.open(input_patch_file, 'w', **profile) as dst:
            dst.write(input_patch)
        
        profile = output_profile.copy()
        profile.update(width=window_size, height=window_size, count=output_patch.shape[0])
        with rasterio.open(output_patch_file, 'w', **profile) as dst:
            dst.write(output_patch)
        
        input_patch_files.append(input_patch_file)
        output_patch_files.append(output_patch_file)

    data_config = {
        "input_means": input_means,
        "input_stds": input_stds,
        "output_means": output_means,
        "output_stds": output_stds,
        "input_patch_files": input_patch_files,
        "output_patch_files": output_patch_files
    }

    with open('patch_data.json', 'w') as f:
        json.dump(data_config, f)


from sklearn.model_selection import train_test_split

def split_data(json_file_path, train_ratio=0.8, random_state=42):
    """
    从JSON文件加载数据并将其分割为训练集和验证集
    
    参数:
    json_file_path (str): 输入JSON文件的路径
    train_ratio (float): 训练集所占比例，默认为0.8
    random_state (int): 随机种子，用于确保分割的可重复性
    
    返回:
    tuple: 包含训练集和验证集数据的元组
    """
    # 读取JSON文件
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # 提取文件列表
    input_files = data["input_patch_files"]
    output_files = data["output_patch_files"]
    
    # 确保两个文件列表长度相同
    assert len(input_files) == len(output_files), "输入和输出文件列表长度不一致！"
    
    # 生成索引列表
    indices = list(range(len(input_files)))
    
    # 分割索引
    train_indices, val_indices = train_test_split(
        indices, train_size=train_ratio, random_state=random_state
    )
    
    # 创建训练集数据
    train_data = {
        "input_means": data["input_means"],
        "input_stds": data["input_stds"],
        "output_means": data["output_means"],
        "output_stds": data["output_stds"],
        "input_patch_files": [input_files[i] for i in train_indices],
        "output_patch_files": [output_files[i] for i in train_indices]
    }
    
    # 创建验证集数据
    val_data = {
        "input_means": data["input_means"],
        "input_stds": data["input_stds"],
        "output_means": data["output_means"],
        "output_stds": data["output_stds"],
        "input_patch_files": [input_files[i] for i in val_indices],
        "output_patch_files": [output_files[i] for i in val_indices]
    }
    
    return train_data, val_data

def save_to_json(data, output_file_path):
    """
    将数据保存为JSON文件
    
    参数:
    data (dict): 要保存的数据
    output_file_path (str): 输出JSON文件的路径
    """
    with open(output_file_path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    train_data, val_data = split_data(
        'patch_data.json', 
        train_ratio=0.8
    )
    
    # 保存分割结果
    save_to_json(train_data, 'train_patch_data.json')
    save_to_json(val_data, 'eval_patch_data.json')


# # Create DataLoader
# dataset = GeoDataset(input_patch_files, output_patch_files, input_means, output_means)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Example usage
# for input_batch, output_batch in dataloader:
#     print(f"Input batch shape: {input_batch.shape}, Output batch shape: {output_batch.shape}")
#     break