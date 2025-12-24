import numpy as np
import rasterio
from rasterio.transform import from_origin
import os
import glob
import rasterio
import numpy as np
from collections import defaultdict

# --- 从readme.pdf中提取的常量 ---
N_ROWS = 21600  # 纬度行数 
N_COLS = 43200  # 经度列数 
DATA_TYPE = np.float64  # 8字节双精度浮点数 
MISSING_VALUE = -1.0e36  # 缺失值 
NUM_LAYERS = 8 # 假设使用8层模型，如CoLM/CLM 

# --- 地理空间信息 ---
# 范围: 180W, 90N, 180E, 90S 
# 左上角坐标 (经度, 纬度)
UPPER_LEFT_X = -180.0
UPPER_LEFT_Y = 90.0
# 像元分辨率 (30" = 1/120 度)
PIXEL_SIZE = 30.0 / 3600.0

def process_variable_to_geotiff(base_dir, variable_name, num_layers, output_filename):
    """
    读取一个变量的所有分层二进制文件，并将它们合并成一个多波段GeoTIFF。

    参数:
    - base_dir (str): 存放二进制文件的主目录。
    - variable_name (str): 变量的文件名基础部分 (例如 'vf_gravels_s_').
    - num_layers (int): 该变量的图层数。
    - output_filename (str): 输出的GeoTIFF文件名。
    """
    all_layers_data = []
    
    print(f"开始处理变量: {variable_name}")

    # 循环读取每个图层的文件 [cite: 19, 39]
    for i in range(1, num_layers + 1):
        # 根据文档中的命名方式构建文件名，例如 'theta_s_l1' [cite: 42]
        # 您的结构可能是 'vf_gravels_s_8L/grid_xxx/xxx'，需要根据实际情况调整
        # 这里我们假设文件命名为 variable_name + 'l' + layer_number
        file_path = os.path.join(base_dir, f"{variable_name}l{i}")
        
        if not os.path.exists(file_path):
            print(f"警告: 文件未找到 {file_path}, 跳过。")
            continue

        print(f"  正在读取图层 {i}: {file_path}")
        
        # 读取二进制文件 [cite: 22, 23]
        layer_data = np.fromfile(file_path, dtype=DATA_TYPE).reshape((N_ROWS, N_COLS))
        all_layers_data.append(layer_data)

    if not all_layers_data:
        print(f"错误: 未能读取到变量 {variable_name} 的任何数据。")
        return

    # 将所有图层堆叠成一个3D数组 (波段, 高, 宽)
    stacked_data = np.stack(all_layers_data, axis=0)

    # --- 创建GeoTIFF文件 ---
    # 定义地理变换参数 [cite: 10, 21, 22]
    # (x像元大小, x旋转, x左上角, y旋转, y像元大小, y左上角)
    transform = from_origin(UPPER_LEFT_X, UPPER_LEFT_Y, PIXEL_SIZE, PIXEL_SIZE)

    # 定义GeoTIFF的元数据
    profile = {
        'driver': 'GTiff',
        'dtype': DATA_TYPE,
        'nodata': MISSING_VALUE,
        'width': N_COLS,
        'height': N_ROWS,
        'count': num_layers,  # 波段数等于图层数
        'crs': 'EPSG:4326',  # WGS84坐标系
        'transform': transform,
        'BIGTIFF' : "YES"
    }

    # 写入文件
    print(f"正在写入到: {output_filename}")
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(stacked_data)
        # 为每个波段设置描述
        for i in range(num_layers):
            dst.set_band_description(i + 1, f"Layer {i+1}")

    print("处理完成！")


def process_layer_to_geotiff(base_dir, variable_name, layer_number, output_filename):
    """
    读取一个变量的单个图层二进制文件，并将其转换为一个单波段GeoTIFF文件。

    参数:
    - base_dir (str): 存放二进制文件的主目录。
    - variable_name (str): 变量的文件名基础部分 (例如 'vf_gravels_s_').
    - layer_number (int): 图层编号 (1 to NUM_LAYERS)。
    - output_filename (str): 输出的GeoTIFF文件名。
    """
    # 构建文件路径，例如 'vf_gravels_s_l1'
    file_path = os.path.join(base_dir, f"{variable_name}l{layer_number}")
    
    if not os.path.exists(file_path):
        print(f"警告: 文件未找到 {file_path}, 跳过。")
        return

    print(f"正在读取图层 {layer_number}: {file_path}")
    
    # 读取二进制文件
    layer_data = np.fromfile(file_path, dtype=DATA_TYPE).reshape((N_ROWS, N_COLS))

    # --- 创建GeoTIFF文件 ---
    # 定义地理变换参数
    transform = from_origin(UPPER_LEFT_X, UPPER_LEFT_Y, PIXEL_SIZE, PIXEL_SIZE)

    # 定义GeoTIFF的元数据
    profile = {
        'driver': 'GTiff',
        'dtype': DATA_TYPE,
        'nodata': MISSING_VALUE,
        'width': N_COLS,
        'height': N_ROWS,
        'count': 1,  # 单波段
        'crs': 'EPSG:4326',  # WGS84坐标系
        'transform': transform,
        'BIGTIFF': 'YES'  # 启用BigTIFF以支持大文件
    }

    # 写入文件
    print(f"正在写入到: {output_filename}")
    with rasterio.open(output_filename, 'w', **profile) as dst:
        dst.write(layer_data, 1)  # 写入到第一个波段
        dst.set_band_description(1, f"Layer {layer_number}")

    print(f"图层 {layer_number} 处理完成！")

def main1():
    # 数据根目录，请根据您的实际情况修改
    # 您的结构是 'vf_gravels_s_8L/grid_xxx/xxx'，这需要更复杂的路径解析
    # 以下示例假设文件结构是扁平的，例如 'vf_gravels_s_l1', 'vf_gravels_s_l2', ...
    data_root = "./" # 示例目录
    output_dir = "./"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义您想处理的变量列表和它们的层数
    # 例如：(变量名前缀, 层数)
    variables_to_process = [
        ('vf_gravels_s_', NUM_LAYERS),
        ('vf_om_s_', NUM_LAYERS),
        ('vf_quartz_mineral_s_', NUM_LAYERS),
        ('vf_sand_s_', NUM_LAYERS),
        ('vf_silt_s_', NUM_LAYERS),
        ('vf_clay_s_', NUM_LAYERS),
        ('theta_s_', NUM_LAYERS),

    ]

    for var_prefix, num_layers in variables_to_process:
        # 调整此处的逻辑以匹配您的 'vf_gravels_s_8L/grid_xxx/xxx' 结构
        # 例如，您可能需要用 os.walk 来查找文件
        
        # 假设文件在 data_root 中，且命名为 vf_gravels_s_l1, l2, ...
        # 如果您的文件都在一个文件夹里，可以这样调用：
        input_data_path = os.path.join(data_root, var_prefix+'8L') # 假设二进制文件都在这里
        output_file = os.path.join(output_dir, f"{var_prefix.strip('_')}.tif")
        
        process_variable_to_geotiff(input_data_path, var_prefix, num_layers, output_file)


def main():
    # 数据根目录，请根据您的实际情况修改
    data_root = "./"  # 示例目录
    output_dir = "./output"  # 输出目录
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 定义您想处理的变量列表和它们的层数
    variables_to_process = [
        ('vf_gravels_s_', NUM_LAYERS),
        ('vf_om_s_', NUM_LAYERS),
        ('vf_quartz_mineral_s_', NUM_LAYERS),
        ('vf_sand_s_', NUM_LAYERS),
        ('vf_silt_s_', NUM_LAYERS),
        ('vf_clay_s_', NUM_LAYERS),
        ('theta_s_', NUM_LAYERS),
    ]

    for var_prefix, num_layers in variables_to_process:
        # 假设文件在 data_root 中，例如 'vf_gravels_s_8L/vf_gravels_s_l1'
        input_data_path = os.path.join(data_root, var_prefix + '8L')
        
        for layer in range(1, num_layers + 1):
            output_file = os.path.join(output_dir, f"{var_prefix.strip('_')}_l{layer}.tif")
            process_layer_to_geotiff(input_data_path, var_prefix, layer, output_file)


def merge_layers():
    # Directory containing the input GeoTIFF files
    input_dir = "/nfs-data3/zlx/land/LandAI/output"  # Modify this to your actual input directory
    output_dir = "/nfs-data3/zlx/land/LandAI/output/merged_layers"  # Directory for output merged GeoTIFF files

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Step 1: Group files by layer suffix (e.g., l1, l2, ..., l8)
    layer_files = defaultdict(list)
    for file_path in glob.glob(os.path.join(input_dir, "*.tif")):
        # Extract the layer suffix (e.g., l1, l2) from the filename
        filename = os.path.basename(file_path)
        if filename.endswith(".tif"):
            layer_suffix = filename.split("_")[-1].replace(".tif", "")
            if layer_suffix.startswith("l") and layer_suffix[1:].isdigit():
                layer_files[layer_suffix].append(file_path)

    # Step 2: Process each layer group
    for layer, files in layer_files.items():
        if not files:
            print(f"No files found for layer {layer}, skipping.")
            continue

        print(f"Processing layer {layer} with {len(files)} files: {files}")

        # Read the first file to get metadata and shape
        with rasterio.open(files[0]) as src:
            base_profile = src.profile.copy()
            height, width = src.height, src.width
            dtype = src.dtypes[0]
            nodata = src.nodata if src.nodata is not None else -1.0e36  # Default nodata value

        # Initialize array to store all bands for this layer
        all_bands = []

        # Step 3: Read each file and stack its band
        for file_path in files:
            with rasterio.open(file_path) as src:
                # Verify dimensions match
                if src.height != height or src.width != width:
                    print(f"Warning: File {file_path} has mismatched dimensions, skipping.")
                    continue
                # Read the single band
                band_data = src.read(1)  # Read the first (and only) band
                all_bands.append(band_data)

        if not all_bands:
            print(f"No valid data for layer {layer}, skipping.")
            continue

        # Stack bands into a 3D array (bands, height, width)
        stacked_data = np.stack(all_bands, axis=0)

        # Step 4: Create output multi-band GeoTIFF
        output_file = os.path.join(output_dir, f"merged_layer_{layer}.tif")
        profile = base_profile.copy()
        profile.update(
            count=len(all_bands),  # Number of bands
            dtype=dtype,
            nodata=nodata,
            BIGTIFF='YES'  # Enable BigTIFF for large files
        )

        print(f"Writing merged GeoTIFF for layer {layer} to {output_file}")
        with rasterio.open(output_file, 'w', **profile) as dst:
            dst.write(stacked_data)
            # Set band descriptions based on input file names
            for i, file_path in enumerate(files, 1):
                band_name = os.path.basename(file_path).replace(f"_{layer}.tif", "")
                dst.set_band_description(i, band_name)

        print(f"Layer {layer} processing completed!")

    print("All layers processed.")

if __name__ == "__main__":
    # main()
    merge_layers()
