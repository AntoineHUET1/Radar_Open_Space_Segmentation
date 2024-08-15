import matplotlib.pyplot as plt
from scipy import interpolate
from natsort import natsorted
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import numpy as np
import os
import json
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator

def Load_Radar_Data(radar_path, cfg):
    if cfg.Merge_Radar_images == 1:
        data_arrays = []
        for file in radar_path:
            data_arrays.append(np.load(file))

        stacked_data = np.stack(data_arrays, axis=0)
        radar_data = np.mean(stacked_data, axis=0)

        radar_data = np.flipud(radar_data)
        radar_data = radar_data.reshape(256, 256, 1)
        if cfg.Radar_Range <= 25:
            radar_data = radar_data[128:, :, :]
        radar_data = np.where(np.isnan(radar_data), 0, radar_data)
        return radar_data

    else:
        radar_data = np.load(radar_path)
        radar_data = np.flipud(radar_data)
        radar_data = radar_data.reshape(256, 256, 1)
        if cfg.Radar_Range <= 25:
            radar_data = radar_data[128:, :, :]
        radar_data = np.where(np.isnan(radar_data), 0, radar_data)
        return radar_data


def Load_GT_Data(gt_path, cfg):
    # Parameters:

    Type = 0  # 0: min value , 1: mean calue

    Gt_data = np.load(gt_path)

    # Standardize the data
    data_standardized = StandardScaler().fit_transform(Gt_data)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.15, min_samples=3)
    labels = dbscan.fit_predict(data_standardized)

    # Keep only points labeled as part of a cluster (exclude outliers)
    keep_points = labels != -1
    filtered_data = Gt_data[keep_points]

    Scaled_Data = []
    Have_GT = []
    for i in range(cfg.GT_Output_shape[0]):
        keep = np.where((filtered_data[:, 0] + cfg.FOV / 2 >= i * cfg.FOV / (cfg.GT_Output_shape[0] + 1)) & (
                    filtered_data[:, 0] + cfg.FOV / 2 < (i + 1) * cfg.FOV / (cfg.GT_Output_shape[0] + 1)))[0]
        if len(keep):
            if Type == 0:
                Scaled_Data.append(np.min(filtered_data[keep, 1]))
                Have_GT.append(1)
            else:
                Scaled_Data.append(np.mean(filtered_data[keep, 1]))
                Have_GT.append(1)
        else:
            Scaled_Data.append(49.9)
            Have_GT.append(0)

    Have_GT = np.array(Have_GT).reshape(cfg.GT_Output_shape[0])
    GT_Output_data = np.array(Scaled_Data).reshape(cfg.GT_Output_shape[0])


    # Remove all values above 25m:
    Have_GT[GT_Output_data > cfg.Radar_Range] = 0
    GT_Output_data[GT_Output_data > cfg.Radar_Range] = cfg.Radar_Range
    GT_Output_data = GT_Output_data * 50 / cfg.Radar_Range

    #GT_Output_data = np.clip(GT_Output_data, 0.51, 48.49)
    GT_Output_data = np.clip(GT_Output_data, 0, 49)
    # Take closest integer value
    GT_Output_data = np.round(GT_Output_data)
    GT = np.stack((Have_GT, GT_Output_data), axis=1)
    return GT


def load_sequence_data(sequence_path,cfg):
    # data path
    radar_data_path = os.path.join(sequence_path, 'Radar_Data')
    gt_path = os.path.join(sequence_path, 'GT')

    # sort the files
    radar_data_files = natsorted(os.listdir(radar_data_path))
    gt_files = natsorted(os.listdir(gt_path))

    if cfg.Merge_Radar_images == 1:
        radar_data = []
        gt_data = []
        for idx in range(len(radar_data_files)):
            # Merge the data files if they are consecutive:
            List_of_merged_radars = []
            int_list = [int(s.split('.')[0]) for s in radar_data_files]
            if idx > 0:
                if int_list[idx - 1] == int_list[idx] - 1:
                    List_of_merged_radars.append(radar_data_files[idx - 1])
            List_of_merged_radars.append(radar_data_files[idx])
            if idx < len(radar_data_files) - 1:
                if int_list[idx + 1] == int_list[idx] + 1:
                    List_of_merged_radars.append(radar_data_files[idx + 1])
            # Add GT data:
            gt_data.append(Load_GT_Data(os.path.join(gt_path, gt_files[idx]), cfg))
            # Load the radar data:
            radar_data.append(Load_Radar_Data([os.path.join(radar_data_path, file) for file in List_of_merged_radars], cfg))

    else:
        # load the data
        radar_data = [Load_Radar_Data(os.path.join(radar_data_path, file), cfg) for file in radar_data_files]
        gt_data = [Load_GT_Data(os.path.join(gt_path, file), cfg) for file in gt_files]

    if cfg.Radar_Range not in [25, 50]:

        if cfg.Radar_Range > 25:
            Factor=50
        else:
            Factor=25

        # Original dimensions
        real_x_dim = int(cfg.Radar_Range * radar_data[0].shape[0] / Factor)
        # Original grid dimensions
        x_dim = np.linspace(0, real_x_dim - 1, real_x_dim)
        y_dim = np.linspace(0, radar_data[0].shape[1] - 1, radar_data[0].shape[1])

        # Desired new grid dimensions
        new_x_dim = np.linspace(0, real_x_dim - 1, radar_data[0].shape[0])
        new_y_dim = np.linspace(0, radar_data[0].shape[1] - 1, radar_data[0].shape[1])


        for index, Radar_Data in enumerate(radar_data):
            # Interpolating
            Radar_Data = Radar_Data[Radar_Data.shape[0] - real_x_dim:Radar_Data.shape[0], :]

            # Squeeze the last dimension for interpolation, keeping the (x, y) shape
            Radar_Data_squeezed = Radar_Data.squeeze()

            # Create the interpolator function
            interp_func = RegularGridInterpolator((x_dim, y_dim), Radar_Data_squeezed, method='linear')

            # Generate a meshgrid for the new dimensions
            new_x_grid, new_y_grid = np.meshgrid(new_x_dim, new_y_dim, indexing='ij')

            # Interpolate over the new grid
            new_radar_data = interp_func((new_x_grid, new_y_grid))

            # Add the third dimension back
            new_radar_data = new_radar_data[..., np.newaxis]  # shape becomes (new_x, new_y, 1)
            radar_data[index] = new_radar_data

    # if GT_mode == 1 and GT = only out of range obstacles then remove data:
    if cfg.GT_mode == 1:
        radar_data = [radar for radar, gt in zip(radar_data, gt_data) if np.any(gt[:, 0] == 1)]
        gt_data = [gt for gt in gt_data if np.any(gt[:, 0] == 1)]

    return radar_data, gt_data


def create_dataset(sequence_paths,cfg):
    for sequence_path in sequence_paths:
        radar_data, gt_data = load_sequence_data(sequence_path,cfg)

        for radar, gt in zip(radar_data, gt_data):
            yield radar, gt

def count_batches(dataset):
    return sum(1 for _ in dataset)

def create_dataloader(sequence_paths,cfg):
    dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset(sequence_paths, cfg),
        output_signature=(
            tf.TensorSpec(shape=cfg.input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=cfg.GT_Output_shape, dtype=tf.float32)  # Assuming your GT data is of float type
        )
    )

    dataset = dataset.batch(cfg.HP_BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Convert list of paths to a tuple
    sequence_paths_tuple = tuple(sequence_paths)

    cache = load_cache(cfg.CACHE_FILE)

    if json.dumps(sequence_paths_tuple) in cache:
        dataloader_length = cache[json.dumps(sequence_paths_tuple)]
    else:
        dataloader_length = count_batches(dataset)
        cache[json.dumps(sequence_paths_tuple)] = dataloader_length
        save_cache(cache, cfg.CACHE_FILE)

    return dataset, dataloader_length

def Generate_Data(cfg, Train_sequence_paths, Val_sequence_paths, Test_sequence_paths):
    train_dataloader, train_dataloader_length = create_dataloader(Train_sequence_paths, cfg)
    val_dataloader, val_dataloader_length = create_dataloader(Val_sequence_paths, cfg)
    test_dataloader, test_dataloader_length = create_dataloader(Test_sequence_paths, cfg)
    return train_dataloader, train_dataloader_length, val_dataloader, val_dataloader_length, test_dataloader, test_dataloader_length

def serialize_cache(cache):
    # Convert dictionary keys to strings
    return {json.dumps(key): value for key, value in cache.items()}

def deserialize_cache(serialized_cache):
    # Convert dictionary keys back to tuples
    return {json.loads(key): value for key, value in serialized_cache.items()}

def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            serialized_cache = json.load(f)
            return deserialize_cache(serialized_cache)
    return {}

def save_cache(cache, cache_file):
    serialized_cache = serialize_cache(cache)
    with open(cache_file, 'w') as f:
        json.dump(serialized_cache, f)

def format_value(value):
    """Format the value for Python source code representation."""
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, list):
        return f'[{", ".join(format_value(v) for v in value)}]'
    elif isinstance(value, tuple):
        return f'({", ".join(format_value(v) for v in value)})'
    elif isinstance(value, dict):
        return f'{{{", ".join(f"{format_value(k)}: {format_value(v)}" for k, v in value.items())}}}'
    elif isinstance(value, bool):
        return 'True' if value else 'False'
    elif isinstance(value, (int, float)):
        return str(value)
    else:
        raise TypeError(f'Unsupported type: {type(value)}')

def save_config(cfg, config_path):
    """Save the configuration to a Python file."""
    with open(config_path, 'w') as f:
        for attr in dir(cfg):
            if not attr.startswith('__'):
                value = getattr(cfg, attr)
                formatted_value = format_value(value)
                f.write(f"{attr} = {formatted_value}\n")