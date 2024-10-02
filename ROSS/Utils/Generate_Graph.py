import matplotlib.pyplot as plt
import tensorflow as tf

from ..Model.Model_ROSS import build_ROSS_32_50
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
from scipy import stats
import platform
import subprocess
from .Data_Loaders import prediction

Transform_matrix =np.array([[ 6.06826773e-03,-9.99947570e-01,8.24823480e-03,-1.08599340e-03],
                   [-2.02354674e-05,-8.24850946e-03,-9.99965980e-01,-1.02729296e-01],
                   [ 9.99981588e-01, 6.06789438e-03,-7.02885703e-05,-1.39408232e-01],
                   [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

CLASSES = ['pedestrian', 'deformed pedestrian', 'bicycle', 'car', 'van', 'bus', 'truck',
           'motorcycle', 'stop sign', 'traffic light', 'traffic sign', 'traffic cone', 'fire hydrant',
           'guard rail', 'pole', 'pole group', 'road', 'sidewalk', 'wall', 'building', 'vegetation',
           'terrain',
           'ground', 'crosstalk', 'noise', 'others', 'animal', 'unpainted', 'cyclist', 'motorcyclist',
           'unclassified vehicle', 'obstacle', 'trailer', 'barrier', 'bicycle rack', 'construction vehicle','??','Unknown']


COLOR = [(176, 242, 182), (9, 82, 40), (255, 127, 0), (119, 181, 254), (15, 5, 107), (206, 206, 206),
         (91, 60, 17), (88, 41, 0), (217, 33, 33), (255, 215, 0), (48, 25, 212), (230, 110, 60),
         (240, 0, 32), (140, 120, 130), (80, 120, 130), (80, 120, 180), (255, 0, 0), (30, 70, 30),
         (230, 230, 130), (230, 130, 130), (60, 250, 60), (100, 140, 40), (100, 40, 40), (250, 10, 10),
         (250, 250, 250), (128, 128, 128), (250, 250, 10), (255, 255, 255), (198, 238, 242),
         (100, 152, 255),
         (50, 130, 200), (100, 200, 50), (255, 150, 120), (100, 190, 240), (20, 90, 200), (80, 40, 0),
         (128, 128, 128), (0, 0, 0)]

# Cylinder projection parameters
image_width = 2000
image_height = 500
FOV_width = 3.6651914291880923
FOV_height = 1.1780972450961724


def calculate_iou_vectorized(boxes1, boxes2, cfg):
    # Calculate width and height for all boxes at once
    w = cfg.FOV / cfg.GT_Output_shape[0]

    # Heights (boxes are 1D in your case)
    h1 = boxes1
    h2 = boxes2

    # Intersect height and width for all pairs
    intersect_w = np.clip(w - 0, 0, None)  # Width is constant
    intersect_h = np.clip(np.minimum(h1, h2), 0, None)  # Clip to ensure non-negative intersection height

    # Calculate intersection area
    intersect_area = intersect_w * intersect_h

    # Calculate area of each box
    box1_area = w * h1
    box2_area = w * h2

    # Calculate IoU
    iou = intersect_area / (box1_area + box2_area - intersect_area)

    return iou


def open_images(image_paths):
    for image_path in image_paths:
        # Normalize the path to handle different OS
        image_path = os.path.abspath(image_path)

        # Determine the OS and use the appropriate command
        if platform.system() == 'Windows':
            subprocess.run(['start', image_path], shell=True)
        elif platform.system() == 'Darwin':  # macOS
            subprocess.run(['open', image_path])
        elif platform.system() == 'Linux':
            subprocess.run(['xdg-open', image_path])
        else:
            raise OSError('Unsupported operating system.')


def convert_difficulty(list_Angle_Difficulty, cfg):
    # Calculate step value based on cfg.FOV and cfg.GT_Output_shape
    step = cfg.FOV / cfg.GT_Output_shape[0]

    # Loop through the list of angle difficulties
    for i in range(len(list_Angle_Difficulty)):
        value = list_Angle_Difficulty[i]

        # Map the values to the difficulty levels
        if value == step:
            list_Angle_Difficulty[i] = 5
        elif value == 2 * step:
            list_Angle_Difficulty[i] = 4
        elif value in [3 * step, 4 * step]:
            list_Angle_Difficulty[i] = 3
        elif value in [5 * step, 6 * step]:
            list_Angle_Difficulty[i] = 2
        elif value > 7 * step:
            list_Angle_Difficulty[i] = 1

    return list_Angle_Difficulty


def genrerat_Graph(checkpoint_path, Data, cfg, label, Save_fig=False, Show_fig=False):

    step = cfg.FOV / cfg.GT_Output_shape[0]

    Fontsize = 20
    # ==================== Default Val ====================
    Thresholds = [0, 0.2, 0.4, 0.6, 0.8]
    Difficulties_val = [5, 4, 3, 2, 1]
    Difficulties_val_label = ['Obstacle size =<'+str(step*1)+'°','Obstacle size >='+str(step*2)+'° and <= '+str(step*3)+'°','Obstacle size >='+str(step*4)+'° and <= '+str(step*5)+'°','Obstacle size >='+str(step*6)+'° and <= '+str(step*7)+'°','Obstacle size >='+str(step*8)+'°']
    colors = [(204, 51, 51), (255, 106, 0), (255, 204, 0), (212, 255, 0), (122, 221, 122)]

    Distance_treshold = [i * cfg.Radar_Range / 5 for i in range(1, 6)]
    directory_path = os.path.dirname(checkpoint_path)

    # Create the graph directory if it does not exist:
    if not os.path.exists(directory_path + '/Graphs'):
        os.makedirs(directory_path+ '/Graphs')

    if label in ['test', 'train', 'val']:
        save_path = directory_path + '/Graphs/' + label
        # Create the label directory if it does not exist:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    else:

        save_path = directory_path + '/Graphs/' + 'Custom/'+label
        # Create the label directory if it does not exist:
        if not os.path.exists(save_path):
            os.makedirs(save_path)


    # Check if the graph as already been generated
    if (os.path.exists(save_path + '/Error_Distribution.png') and
            os.path.exists(save_path + '/Prediction_confidence_level_MAE.png')
            and os.path.exists(save_path + '/Error_relative_to_distance.png')
            and os.path.exists(save_path + '/Error_relative_to_angle.png')):
        print(label + ' graphs already generated')
    else:
        # ==================== Parameters ====================
        Factor = cfg.Radar_Range / cfg.Output_vertices

        # ==================== Load Model ====================

        model = build_ROSS_32_50(cfg)

        model.load_weights(checkpoint_path)

        # ==================== Inference ====================
        Pred_Full, GT_Full, Pred_val_Full = prediction(Data,model, cfg)

        # Stack the results into arrays
        Pred_Full = np.stack(Pred_Full, axis=0)
        GT_Full = np.stack(GT_Full, axis=0)
        GT_Full = GT_Full[..., np.newaxis]
        # Create a 0 matrix for the difficulty with the same shape as the GT_Full

        Max_variation=1*cfg.Output_vertices/cfg.Radar_Range

        Difficulty= np.zeros(GT_Full.shape)
        for i in range(GT_Full.shape[0]):
            for j in range(GT_Full.shape[1]):
                Valid_neighbors = 0
                if np.isnan(GT_Full[i, j, 0]):
                    Difficulty[i, j, 0] = np.nan
                else:
                    if j > 0:
                        jL = 1
                        while abs(GT_Full[i, j - jL, 0] - GT_Full[i, j - jL + 1, 0]) <= Max_variation and not np.isnan(
                                GT_Full[i, jL, 0]):
                            Valid_neighbors += 1
                            jL += 1
                            if j - jL < 0:
                                break

                    if j < GT_Full.shape[1] - 1:
                        jR = 1
                        while abs(GT_Full[i, j + jR, 0] - GT_Full[i, j + jR - 1, 0]) <= Max_variation and not np.isnan(
                                GT_Full[i, jR, 0]):
                            Valid_neighbors += 1
                            jR = jR + 1
                            if j + jR > GT_Full.shape[1] - 1:
                                break

                    Difficulty[i, j, 0] = Valid_neighbors + 1

        Angle_Difficulty=Difficulty*cfg.FOV/cfg.GT_Output_shape[0]

        Pred_val_Full = np.stack(Pred_val_Full, axis=0)

        # Flatten the arrays
        flattened_list_Pred = Pred_Full.reshape(-1, Pred_Full.shape[-1])
        flattened_list_GT = GT_Full.reshape(-1, GT_Full.shape[-1])
        flattened_list_Pred_val = Pred_val_Full.reshape(-1, Pred_val_Full.shape[-1])
        flattened_Angle_Difficulty = Angle_Difficulty.reshape(-1, Angle_Difficulty.shape[-1])

        # Convert to lists with transformation and skip NaN values
        list_Pred = [item[0] * Factor for item in flattened_list_Pred if not np.isnan(item[0])]
        list_GT = [item[0] * Factor for item in flattened_list_GT if not np.isnan(item[0])]
        list_Pred_val = [item[0] for item in flattened_list_Pred_val if not np.isnan(item[0])]
        list_Angle_Difficulty = [item[0] for item in flattened_Angle_Difficulty if not np.isnan(item[0])]


        list_Difficulty=convert_difficulty(list_Angle_Difficulty, cfg)

        plt.figure(figsize=(20, 10))
        # Iterate over each threshold

        list_percentage_GT = []
        list_mae = []
        for i, Threshold in enumerate(Thresholds):
            keep = np.where(np.array(list_Pred_val) > Threshold)[0]

            if len(keep) == 0:
                continue
            # Calculate the error
            mae = mean_absolute_error(np.array(list_Pred)[keep], np.array(list_GT)[keep])

            error_values = np.array(list_Pred)[keep] - np.array(list_GT)[keep]

            list_percentage_GT.append(round(100 * len(keep) / len(list_GT), 2))
            list_mae.append(round(mae, 2))

            bound = (cfg.Radar_Range // cfg.Radar_Range_resolution + 1) * cfg.Radar_Range_resolution

            bins = np.arange(-bound, bound + cfg.Radar_Range_resolution, cfg.Radar_Range_resolution)


            # Create histogram
            hist, bins = np.histogram(error_values, bins=bins)

            # Calculate percentage
            percentage = (hist / len(list_Pred)) * 100

            # Color

            color = [c / 255 for c in colors[i]]

            # Plot histogram with percentage
            plt.bar(bins[:-1], percentage, width=(bins[1] - bins[0]), color=color, label=f'P>{Threshold}')


        # Customize labels and title
        plt.xlabel('Error Values (m)', fontsize=Fontsize)
        plt.ylabel('Percentage', fontsize=Fontsize)
        plt.title('Distribution of Error Values (%) for ' + label + ' data', fontsize=1.5 * Fontsize)

        plt.legend()

        # Show plot
        if Save_fig == True:
            plt.savefig(save_path + '/Error_Distribution.png')

        plt.figure(figsize=(20, 10))
        list_percentage_GT = []
        list_mae = []
        for i, Threshold in enumerate(Difficulties_val):

            keep = np.where(np.array(list_Difficulty) <= Threshold)[0]

            if len(keep) == 0:
                continue

            IOU = calculate_iou_vectorized(np.array(list_GT)[keep], np.array(list_Pred)[keep], cfg)

            # Calculate average IoU
            average_iou = np.mean(IOU)


            # save the average iou in a file
            with open(save_path + '/average_iou.txt', 'a') as f:
                # append
                f.write(f'Average IoU for difficulty <= {Threshold} is {average_iou:.2f}\n')

            # Calculate the error
            mae = mean_absolute_error(np.array(list_Pred)[keep], np.array(list_GT)[keep])

            error_values = np.array(list_Pred)[keep] - np.array(list_GT)[keep]

            list_percentage_GT.append(round(100 * len(keep) / len(list_GT), 2))
            list_mae.append(round(mae, 2))

            bound = (cfg.Radar_Range // cfg.Radar_Range_resolution + 1) * cfg.Radar_Range_resolution

            bins = np.arange(-bound, bound + cfg.Radar_Range_resolution, cfg.Radar_Range_resolution)

            # Create histogram
            hist, bins = np.histogram(error_values, bins=bins)

            # Calculate percentage
            percentage = (hist / len(list_Pred)) * 100

            # Color

            color = [c / 255 for c in colors[i]]

            # Plot histogram with percentage
            plt.bar(bins[:-1], percentage, width=(bins[1] - bins[0]), color=color, label=Difficulties_val_label[i])

        # Customize labels and title
        plt.xlabel('Error Values (m)', fontsize=Fontsize)
        plt.ylabel('Percentage', fontsize=Fontsize)
        plt.title('Distribution of Error Values (%) for ' + label + ' data', fontsize=1.5 * Fontsize)

        plt.legend()

        # Show plot
        if Save_fig == True:
            plt.savefig(save_path + '/Error_Distribution_difficulty.png')


        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(15, 10))

        for i in range(len(list_mae)):
            ax.barh(i, list_mae[i], color=np.array(colors[i]) / 255, edgecolor='black')
            ax.text(list_mae[i] + 0.01, i, f'{list_mae[i]:.2f}', va='center', color='black', fontsize=Fontsize)

        # Customize plot
        ax.set_yticks(range(len(Thresholds)))

        if Thresholds[0] == 0:
            ax.set_yticklabels(['All data' if t == 0 else f'Threshold > {t}' for t in Thresholds], fontsize=Fontsize)
        else:
            ax.set_yticklabels([f'Threshold > {t}' for t in Thresholds], fontsize=Fontsize)

        ax.set_xlim(0, 4)

        ax.set_xlabel('MAE', fontsize=Fontsize)
        ax.set_title(
            'Mean Absolute Error (MAE) for Predictions within Specific Prediction Confidence Levels (Data ' + label + ')',
            fontsize=Fontsize * 0.8)

        # Show plot
        plt.tight_layout()

        # Save or show plot
        if Save_fig:
            plt.savefig(save_path + '/Prediction_confidence_level_MAE.png')



        # ============ Error relative to the distance ============
        # Calculate the error relative to the distance
        mae_list = []
        median_list = []

        # Convert to numpy arrays for easier manipulation
        list_GT = np.array(list_GT)
        list_Pred = np.array(list_Pred)
        list_Percentage_Perfect = []

        for i in range(len(Distance_treshold)):
            lower_bound = Distance_treshold[i] - Distance_treshold[0]
            upper_bound = Distance_treshold[i]

            # Filter based on range
            keep = np.where((list_GT > lower_bound) & (list_GT < upper_bound))[0]

            if len(keep) == 0:
                mae_list.append(np.nan)
                median_list.append(np.nan)
                list_Percentage_Perfect.append(np.nan)
                continue

            # Calculate the MAE
            mae = mean_absolute_error(list_Pred[keep], list_GT[keep])
            mae_list.append(mae)

            # Calculate the median of absolute errors
            median = np.median(np.abs(list_Pred[keep] - list_GT[keep]))
            median_list.append(median)

            # Calculate the percentage of perfect predictions
            percentage_perfect = len(np.where(np.abs(list_Pred[keep] - list_GT[keep]) < cfg.Radar_Range/cfg.Output_vertices)[0]) / len(keep)
            percentage_perfect = f'{round(100 * percentage_perfect, 2)}%'
            list_Percentage_Perfect.append(percentage_perfect)

        # Create bar plot
        x = np.arange(len(Distance_treshold))  # The x locations for the groups

        plt.figure(figsize=(15, 10))

        # Plot MAE bars
        plt.bar(x, mae_list, width=0.4, color='skyblue', label='MAE', align='center')
        # Plot Median bars
        plt.bar(x, median_list, width=0.4, color='salmon', label='Median Error', align='center')

        # Add labels for ranges
        for i in range(len(Distance_treshold)):
            plt.text(i, max(mae_list[i], median_list[i]) + 0.1, list_Percentage_Perfect[i], ha='center', va='bottom',
                     fontsize=Fontsize)

        plt.xlabel('Threshold Index (m)', fontsize=Fontsize)
        plt.ylabel('Error (m)', fontsize=Fontsize)
        plt.title('Error Relative to Distance Range with % of perfect prediction', fontsize=Fontsize)
        plt.xticks(x, [f'{Distance_treshold[i] - Distance_treshold[0]:.1f} - {Distance_treshold[i]:.1f}' for i in
                       range(len(Distance_treshold))])
        plt.legend()
        plt.grid(axis='y', linestyle='--')
        if Save_fig:
            plt.savefig(save_path + '/Error_relative_to_distance.png')

        # ============ Error relative to the angle ============
        # Calculate the error relative to the angle
        mae_list = []
        median_list = []
        Angle_List = []

        for i in range(Pred_Full.shape[1]):
            Angle = i * cfg.FOV / Pred_Full.shape[1] - cfg.FOV / 2 + cfg.FOV / Pred_Full.shape[1] / 2
            # Filter based on angle
            Angle_GT = GT_Full[:, i, 0] * Factor
            Angle_Pred = Pred_Full[:, i, 0] * Factor
            # Remove NaN values
            keep = np.where(~np.isnan(Angle_GT))[0]
            Angle_GT = Angle_GT[keep]
            Angle_Pred = Angle_Pred[keep]

            mae = mean_absolute_error(Angle_Pred, Angle_GT)
            mae_list.append(mae)

            median = np.median(np.abs(Angle_Pred - Angle_GT))
            median_list.append(median)

            Angle_List.append(Angle)

        plt.figure(figsize=(15, 10))

        # Plot MAE line
        plt.plot(Angle_List, mae_list, marker='o', color='skyblue', label='MAE', linestyle='-', linewidth=3)

        # Plot Median line
        plt.plot(Angle_List, median_list, marker='o', color='salmon', label='Median Error', linestyle='-', linewidth=3)

        # Add labels for values
        for i in range(len(Angle_List)):
            plt.text(Angle_List[i], mae_list[i] + 0.2, f'{mae_list[i]:.2f}', ha='center', va='bottom', fontsize=10,
                     color='blue')
            plt.text(Angle_List[i], median_list[i] + 0.2, f'{median_list[i]:.1f}', ha='center', va='bottom', fontsize=10,
                     color='red')

        plt.xlabel('Angle (degrees)', fontsize=14)
        plt.ylabel('Error ', fontsize=14)
        plt.title('Error Relative to Angle', fontsize=16)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(save_path + '/Error_relative_to_angle.png')

    if Show_fig:
        open_images([save_path + '/Error_Distribution.png',
                     save_path + '/Prediction_confidence_level_MAE.png',
                     save_path + '/Error_relative_to_distance.png',
                     save_path + '/Error_relative_to_angle.png'])

