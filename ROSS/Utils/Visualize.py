import os
import random
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from .Data_Loaders import Load_GT_Data,Load_Radar_Data,adjust_radar_data_based_on_range_and_gt
import ROSS.cfg.ROSS_Config as cfg

def get_files(path):
    if path is not None:
        if os.path.isdir(path):
            files = natsorted(os.listdir(path))
            return [os.path.join(path, file) for file in files]
        else:
            return [path]
    return []

def visualisation_plot(Radar_data_path,Frontal_Image_path,GT_path,fps,GT_point_cloud):

    if Radar_data_path is None and Frontal_Image_path is None and GT_path is None:
        print('No data to visualize')
        return

    Radar_File = get_files(Radar_data_path)
    Frontal_Image = get_files(Frontal_Image_path)
    GT_File = get_files(GT_path)

    if Frontal_Image and Radar_File:
        plt.figure(figsize=(10, 8))
    else:
        if Frontal_Image:
            plt.figure(figsize=(10, 4))
        if Radar_File:
            plt.figure(figsize=(10, 4))

    for i in range(len(GT_File)):
        if Frontal_Image and Radar_File:
            # Both camera and radar are available
            plt.subplot(2, 1, 1)  # 2 rows, 1 column, 1st plot
            image = plt.imread(Frontal_Image[i])
            plt.imshow(image)
            plt.title('Frontal Camera Image')
            plt.subplot(2, 1, 2)  # 2 rows, 1 column, 2nd plot
            Radar_data = Load_Radar_Data(Radar_File[i], cfg,Mode_Visualisation=True)
            if GT_point_cloud:
                plt.imshow(Radar_data, extent=[-cfg.FOV/2, cfg.FOV/2, 0, cfg.Radar_Range], aspect='equal')
                plt.title('Radar Data')
                GT = np.load(GT_File[i])
                X=GT[:, 0]
                Y=GT[:, 1]

                plt.scatter(X, Y, c='r', s=4)
                plt.xlim(-cfg.FOV/2, cfg.FOV/2)
                plt.ylim(0, cfg.Radar_Range)
            else:
                gt_data=Load_GT_Data(GT_File[i], cfg)
                radar_data, gt_data=adjust_radar_data_based_on_range_and_gt([Radar_data], [gt_data], cfg)
                plt.imshow(radar_data[0], extent=[-cfg.FOV/2, cfg.FOV/2, 0, cfg.Radar_Range], aspect='equal')
                plt.title('Radar Data')

                GT=gt_data[0][:,1]
                for i in range(cfg.GT_Output_shape[0]):
                   plt.plot([i * cfg.FOV / (cfg.GT_Output_shape[0] + 1) - cfg.FOV/2, (i + 1) * cfg.FOV / (cfg.GT_Output_shape[0] + 1) - cfg.FOV/2],[GT[i]*cfg.Radar_Range/50, GT[i]*cfg.Radar_Range/50], c='r')


            #plt.xlim(-cfg.FOV/2, cfg.FOV/2)
            #plt.ylim(0, cfg.Radar_Range)


        elif Frontal_Image:
            # Only the camera image is available
            image = plt.imread(Frontal_Image[i])
            plt.imshow(image)
            plt.title('Frontal Camera Image')
        elif Radar_File:
            # Only the radar data is available
            Radar_data = Load_Radar_Data(Radar_File[i], cfg)
            plt.imshow(Radar_data, extent=[-60, 60, 0, 50])
            plt.title('Radar Data')

        plt.pause(1/fps)
        plt.clf()



def visualize_data(Sequence=None,No_Frontal_Camera_files=False,No_Radar_files=False,No_GT_files=False,Frame_Number=None,fps=10,GT_point_cloud=False):

    # Parameters:
    Data_path = 'data/ROSS_Dataset'
    Radar_data_path = None
    GT_path = None
    Frontal_Image_path = None

    if Sequence is None:
        # Select a random sequence
        Sequence = random.choice(os.listdir(Data_path))

    Sequence_path = os.path.join(Data_path, Sequence)

    if Frame_Number is not None:
        if not No_Radar_files:
            Radar_data_path = os.path.join(Sequence_path, 'Radar_Data')
            Radar_files = os.listdir(Radar_data_path)
            Radar_files = natsorted(Radar_files)
            Radar_data_path += '/' + Radar_files[Frame_Number]
        if not No_GT_files:
            GT_path = os.path.join(Sequence_path, 'GT') + '/Frame' + str(Frame_Number) + '.npy'
        if not No_Frontal_Camera_files:
            Frontal_Image_path = os.path.join(Sequence_path, 'Resized_images') + '/' + str(Frame_Number) + '.png'
    else:
        if not No_Radar_files:
            Radar_data_path = os.path.join(Sequence_path, 'Radar_Data')
        if not No_GT_files:
            GT_path = os.path.join(Sequence_path, 'GT')
        if not No_Frontal_Camera_files:
            Frontal_Image_path = os.path.join(Sequence_path, 'Resized_images')

    visualisation_plot(Radar_data_path,Frontal_Image_path,GT_path,fps,GT_point_cloud)

    return