from scipy import interpolate
from natsort import natsorted
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2
import tensorflow as tf
import numpy as np
import os


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

        # Original dimensions
        real_x_dim = int(cfg.Radar_Range * radar_data[0].shape[0] / 25)
        x_dim = np.linspace(0, 1, real_x_dim)
        y_dim = np.linspace(0, 1, radar_data[0].shape[1])

        # New dimensions
        new_x = np.linspace(0, 1, radar_data[0].shape[0])
        new_y = np.linspace(0, 1, radar_data[0].shape[1])

        for index, Radar_Data in enumerate(radar_data):
            # Interpolating
            Radar_Data = Radar_Data[Radar_Data.shape[0] - real_x_dim:Radar_Data.shape[0], :]

            interp_func = interpolate.interp2d(y_dim, x_dim, Radar_Data.squeeze(), kind='linear')
            new_Radar_Data = interp_func(new_y, new_x)
            new_Radar_Data = new_Radar_Data[:, :, np.newaxis]
            radar_data[index] = new_Radar_Data

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

    dataloader_length = 0  # total_samples // batch_size
    return dataset, dataloader_length

def Generate_Data(cfg, Train_sequence_paths, Val_sequence_paths, Test_sequence_paths):
    train_dataloader, train_dataloader_length = create_dataloader(Train_sequence_paths, cfg)
    val_dataloader, val_dataloader_length = create_dataloader(Val_sequence_paths, cfg)
    test_dataloader, test_dataloader_length = create_dataloader(Test_sequence_paths, cfg)
    train_dataloader_length, val_dataloader_length, test_dataloader_length = 2 * 523, 2 * 37, 2 * 45
    return train_dataloader, train_dataloader_length, val_dataloader, val_dataloader_length, test_dataloader, test_dataloader_length