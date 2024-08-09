from scipy import interpolate
from natsort import natsorted
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import cv2
import tensorflow as tf
import numpy as np
import os


def Load_Radar_Data(radar_path, Half_length, Merge_Radar_images=0):
    if Merge_Radar_images == 1:
        data_arrays = []
        for file in radar_path:
            data_arrays.append(np.load(file))

        stacked_data = np.stack(data_arrays, axis=0)
        radar_data = np.mean(stacked_data, axis=0)

        radar_data = np.flipud(radar_data)
        radar_data = radar_data.reshape(256, 256, 1)
        if Half_length:
            radar_data = radar_data[128:, :, :]
        radar_data = np.where(np.isnan(radar_data), 0, radar_data)
        return radar_data

    else:
        radar_data = np.load(radar_path)
        radar_data = np.flipud(radar_data)
        radar_data = radar_data.reshape(256, 256, 1)
        if Half_length:
            radar_data = radar_data[128:, :, :]
        radar_data = np.where(np.isnan(radar_data), 0, radar_data)
        return radar_data


def Load_GT_Data(gt_path, GT_Output_shape, Half_length, GT_mode, Mode, FOV, Binary_Camera, Radar_Range):
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
    for i in range(GT_Output_shape[0]):
        keep = np.where((filtered_data[:, 0] + FOV / 2 >= i * FOV / (GT_Output_shape[0] + 1)) & (
                    filtered_data[:, 0] + FOV / 2 < (i + 1) * FOV / (GT_Output_shape[0] + 1)))[0]
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

    Have_GT = np.array(Have_GT).reshape(GT_Output_shape[0])
    GT_Output_data = np.array(Scaled_Data).reshape(GT_Output_shape[0])

    if Half_length and Radar_Range in [25]:
        # Remove all values above 25m:
        Have_GT[GT_Output_data > 25] = 0
        GT_Output_data[GT_Output_data > 25] = 25
        if Mode == 1:
            GT_Output_data = GT_Output_data * 2

    if Half_length and Radar_Range not in [25, 50]:
        # Remove all values above 25m:
        Have_GT[GT_Output_data > Radar_Range] = 0
        GT_Output_data[GT_Output_data > Radar_Range] = Radar_Range
        if Mode == 1:
            GT_Output_data = GT_Output_data * 50 / Radar_Range

    GT_Output_data = np.clip(GT_Output_data, 0.51, 48.49)

    if Binary_Camera:
        # 0 if val < 25, 1 otherwise
        GT_Output_data = np.where(GT_Output_data < 25, 0, 1)

    # plt.plot(GT_Output_data)
    # plt.pause(0.1)
    # plt.clf()
    GT = np.stack((Have_GT, GT_Output_data), axis=1)
    # print(GT.shape)
    # breakpoint()

    return GT


def load_sequence_data(sequence_path, input_shape, GT_Output_shape, Half_length, GT_mode, Mode, Add_Frontal_images, FOV,
                       Merge_Radar_images, Binary_Camera, Radar_Range):
    # data path
    if Merge_Radar_images in [0, 1]:
        radar_data_path = os.path.join(sequence_path, 'Radar_Data')
    else:
        radar_data_path = os.path.join(sequence_path, 'Radar_Data_Avg')

    gt_path = os.path.join(sequence_path, 'GT')

    # sort the files
    radar_data_files = natsorted(os.listdir(radar_data_path))
    gt_files = natsorted(os.listdir(gt_path))

    if Merge_Radar_images == 1:
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
            gt_data.append(
                Load_GT_Data(os.path.join(gt_path, gt_files[idx]), GT_Output_shape, Half_length, GT_mode, Mode, FOV,
                             Binary_Camera, Radar_Range))
            # Load the radar data:
            radar_data.append(
                Load_Radar_Data([os.path.join(radar_data_path, file) for file in List_of_merged_radars], Half_length,
                                Merge_Radar_images))

    else:
        # load the data
        radar_data = [Load_Radar_Data(os.path.join(radar_data_path, file), Half_length) for file in radar_data_files]
        gt_data = [
            Load_GT_Data(os.path.join(gt_path, file), GT_Output_shape, Half_length, GT_mode, Mode, FOV, Binary_Camera,
                         Radar_Range) for file in gt_files]

    if Add_Frontal_images == 0 and Radar_Range not in [25, 50]:

        # Original dimensions
        real_x_dim = int(Radar_Range * radar_data[0].shape[0] / 25)
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

    if Add_Frontal_images != 0:

        # Camera Data:
        if Add_Frontal_images in [1, 4]:
            Frontal_images_path = os.path.join(sequence_path, 'Resized_images')
        if Add_Frontal_images in [2, 5]:
            Frontal_images_path = os.path.join(sequence_path, 'Bird_View')
        if Add_Frontal_images in [3, 6]:
            Frontal_images_path = os.path.join(sequence_path, 'Bird_View_RA')

        images_files = natsorted(os.listdir(Frontal_images_path))

        # Resize the radar data to fit the images shape:
        # Original dimensions
        x_dim = np.linspace(0, 1, radar_data[0].shape[0])
        y_dim = np.linspace(0, 1, radar_data[0].shape[1])

        # New dimensions
        new_x = np.linspace(0, 1, input_shape[0])
        new_y = np.linspace(0, 1, input_shape[1])

        for index, Radar_Data in enumerate(radar_data):
            # Interpolating
            interp_func = interpolate.interp2d(y_dim, x_dim, Radar_Data.squeeze(), kind='linear')
            new_Radar_Data = interp_func(new_y, new_x)
            new_Radar_Data = new_Radar_Data[:, :, np.newaxis]

            # Resize images to fit the radar data shape:
            Image_to_load = cv2.imread(os.path.join(Frontal_images_path, images_files[index]),
                                       cv2.COLOR_BGR2RGB).astype(np.uint8)
            if FOV == 90:
                if Add_Frontal_images in [1, 4]:
                    # Frontal Camera:
                    Image_to_load = Image_to_load[:, 141:1017, :]
                if Add_Frontal_images in [3, 6]:
                    # Bird View RA:
                    Image_to_load = Image_to_load[:, 147:1051, :]
                    # plt.imshow(Image_to_load)
                    # plt.show()
                if Add_Frontal_images in [2, 5]:
                    # Raise an error if Bird View is not supported for FOV=90
                    raise ValueError('Error: Bird View is not supported for FOV=90')

            if Add_Frontal_images in [1, 2, 3]:
                radar_data[index] = np.concatenate(
                    (cv2.resize(Image_to_load, (input_shape[1], input_shape[0])), new_Radar_Data), axis=-1)
            else:
                radar_data[index] = cv2.resize(Image_to_load, (input_shape[1], input_shape[0]))

    # if GT_mode == 1 and GT = only out of range obstacles then remove data:
    if GT_mode == 1:
        radar_data = [radar for radar, gt in zip(radar_data, gt_data) if np.any(gt[:, 0] == 1)]
        gt_data = [gt for gt in gt_data if np.any(gt[:, 0] == 1)]

    return radar_data, gt_data


def create_dataset(sequence_paths, input_shape, GT_Output_shape, Half_length, GT_mode, Mode, Add_Frontal_images, FOV,
                   Merge_Radar_images, Binary_Camera, Radar_Range):
    for sequence_path in sequence_paths:
        radar_data, gt_data = load_sequence_data(sequence_path, input_shape, GT_Output_shape, Half_length, GT_mode,
                                                 Mode, Add_Frontal_images, FOV, Merge_Radar_images, Binary_Camera,
                                                 Radar_Range)

        for radar, gt in zip(radar_data, gt_data):
            yield radar, gt

def create_dataloader(sequence_paths, input_shape, GT_Output_shape, batch_size, Half_length, GT_mode, Mode,
                      Add_Frontal_images, FOV, Merge_Radar_images, Binary_Camera, Radar_Range):
    '''
    total_samples = 0
    for sequence_path in sequence_paths:
        _, gt_data = load_sequence_data(sequence_path, GT_Output_shape, Half_length, GT_mode,Mode,Add_Frontal_images)
        total_samples += len(gt_data)
    '''
    dataset = tf.data.Dataset.from_generator(
        lambda: create_dataset(sequence_paths, input_shape, GT_Output_shape, Half_length, GT_mode, Mode,
                               Add_Frontal_images, FOV, Merge_Radar_images, Binary_Camera, Radar_Range),
        output_signature=(
            tf.TensorSpec(shape=input_shape, dtype=tf.float32),
            tf.TensorSpec(shape=GT_Output_shape, dtype=tf.float32)  # Assuming your GT data is of float type
        )
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    dataloader_length = 0  # total_samples // batch_size
    return dataset, dataloader_length