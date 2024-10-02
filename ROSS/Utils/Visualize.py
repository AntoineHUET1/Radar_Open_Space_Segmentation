import os
import random
import numpy as np
from natsort import natsorted
import matplotlib.pyplot as plt
from .Data_Loaders import Load_GT_Data,Load_Radar_Data,adjust_radar_data_based_on_range_and_gt
import ROSS.cfg.ROSS_Config as cfg


CLASSES = ['pedestrian', 'deformed pedestrian', 'bicycle', 'car', 'van', 'bus', 'truck',
           'motorcycle', 'stop sign', 'traffic light', 'traffic sign', 'traffic cone', 'fire hydrant',
           'guard rail', 'pole', 'pole group', 'road', 'sidewalk', 'wall', 'building', 'vegetation',
           'terrain',
           'ground', 'crosstalk', 'noise', 'others', 'animal', 'unpainted', 'cyclist', 'motorcyclist',
           'unclassified vehicle', 'obstacle', 'trailer', 'barrier', 'bicycle rack', 'construction vehicle','Unknown']

COLOR = [(176, 242, 182), (9, 82, 40), (255, 127, 0), (119, 181, 254), (15, 5, 107), (206, 206, 206),
         (91, 60, 17), (88, 41, 0), (217, 33, 33), (255, 215, 0), (48, 25, 212), (230, 110, 60),
         (240, 0, 32), (140, 120, 130), (80, 120, 130), (80, 120, 180), (30, 30, 30), (30, 70, 30),
         (230, 230, 130), (230, 130, 130), (60, 250, 60), (100, 140, 40), (100, 40, 40), (250, 10, 10),
         (250, 250, 250), (128, 128, 128), (250, 250, 10), (255, 255, 255), (198, 238, 242),
         (100, 152, 255),
         (50, 130, 200), (100, 200, 50), (255, 150, 120), (100, 190, 240), (20, 90, 200), (80, 40, 0),
         (128, 128, 128), (255, 0, 0)]


def Cathesian_to_RA(point_Cloud):

    x = point_Cloud[:, 0]
    y = -point_Cloud[:, 1]
    z = point_Cloud[:, 2]

    #Range:
    r = np.sqrt(x**2 + y**2 + z**2)
    angle = np.arctan2(y, x)
    angle = np.degrees(angle)
    return r, angle,z

def get_files(path):
    if path is not None:
        if os.path.isdir(path):
            files = natsorted(os.listdir(path))
            return [os.path.join(path, file) for file in files]
        else:
            return [path]
    return []

def visualisation_plot(Radar_data_path,Frontal_Image_path,GT_path,fps,GT_point_cloud,GT_Lines,Open_Space):

    if Radar_data_path is None and Frontal_Image_path is None and GT_path is None:
        print('No data to visualize')
        return

    Radar_File = get_files(Radar_data_path)
    Frontal_Image = get_files(Frontal_Image_path)
    GT_File = get_files(GT_path)
    Lines_GT=get_files(os.path.dirname(GT_path)+"/Lines")
    Point_clouds=get_files(os.path.dirname(GT_path)+"/Sorted_Point_Cloud")

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
            # Remove axis
            plt.axis('off')
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
                # add GT legend:
            elif GT_Lines:
                plt.imshow(Radar_data, extent=[-cfg.FOV / 2, cfg.FOV / 2, 0, cfg.Radar_Range], aspect='equal')
                plt.title('Radar Data')
                GT = np.load(Lines_GT[i],allow_pickle=True)
                for i in range(len(GT)):
                    Class=GT[i]['class']
                    Color=COLOR[Class]
                    Lines=GT[i]['line']
                    plt.scatter(Lines[:,0], Lines[:,1], c=np.array(Color).reshape(1, -1) / 255, s=4)
            elif Open_Space:
                plt.subplot(2, 2, 3)
                Point_cloud=np.load(Point_clouds[i],allow_pickle=True)
                for i in range(len(Point_cloud)):
                    Class = Point_cloud[i]['class']
                    Color = COLOR[Class]
                    point_cloud = Point_cloud[i]['point_cloud']
                    plt.scatter(-point_cloud[:, 1], point_cloud[:, 0], c=np.array(Color).reshape(1, -1) / 255, s=4)
                plt.xlim(-cfg.Radar_Range*np.cos(np.pi*(90-cfg.FOV/2)/180),cfg.Radar_Range*np.cos(np.pi*(90-cfg.FOV/2)/180))
                plt.ylim(0, cfg.Radar_Range)
                plt.title('Open Space cartesian')
                plt.xlabel('X (m)')
                plt.ylabel('Y (m)')
                plt.subplot(2, 2, 4)
                for i in range(len(Point_cloud)):
                    Class = Point_cloud[i]['class']
                    Color = COLOR[Class]
                    point_cloud = Point_cloud[i]['point_cloud']
                    Range, Angle, z = Cathesian_to_RA(point_cloud)
                    plt.scatter(Angle, Range, c=np.array(Color).reshape(1, -1) / 255, s=4)


            else:
                gt_data=Load_GT_Data(GT_File[i], cfg)
                radar_data, gt_data=adjust_radar_data_based_on_range_and_gt([Radar_data], [gt_data], cfg)
                plt.imshow(radar_data[0], extent=[-cfg.FOV/2, cfg.FOV/2, 0, cfg.Radar_Range], aspect='equal')
                plt.title('Radar Data')

                GT=gt_data[0][:,1]
                for i in range(cfg.GT_Output_shape[0]):
                   plt.plot([i * cfg.FOV / (cfg.GT_Output_shape[0] + 1) - cfg.FOV/2, (i + 1) * cfg.FOV / (cfg.GT_Output_shape[0] + 1) - cfg.FOV/2],[GT[i]*cfg.Radar_Range/50, GT[i]*cfg.Radar_Range/50], c='r')


            plt.xlim(-cfg.FOV/2, cfg.FOV/2)
            plt.ylim(0, cfg.Radar_Range)
            plt.legend(['Drivable Area limit'], loc='upper right')
            plt.xlabel('Angle (°)')
            plt.ylabel('Range (m)')

            #plt.tight_layout()

        elif Frontal_Image:
            # Only the camera image is available
            image = plt.imread(Frontal_Image[i])
            plt.imshow(image)
            plt.title('Frontal Camera Image')
            plt.axis('off')
        elif Radar_File:
            # Only the radar data is available
            Radar_data = Load_Radar_Data(Radar_File[i], cfg)
            plt.imshow(Radar_data, extent=[-60, 60, 0, 50])
            plt.title('Radar Data')
            plt.legend(['Drivable Area limit'], loc='upper right')
            plt.xlabel('Angle (°)')
            plt.ylabel('Range (m)')

        plt.pause(1/fps)
        plt.clf()



def visualize_data(Sequence=None,No_Frontal_Camera_files=False,No_Radar_files=False,No_GT_files=False,Frame_Number=None,fps=10,GT_point_cloud=False,GT_Lines=False,Open_Space=False):

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


    visualisation_plot(Radar_data_path,Frontal_Image_path,GT_path,fps,GT_point_cloud,GT_Lines,Open_Space)


    return