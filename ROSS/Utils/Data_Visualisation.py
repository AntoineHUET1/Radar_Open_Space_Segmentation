import matplotlib.pyplot as plt
import numpy as np
from natsort import natsorted
import os


def visualize_data(Sequence_path):
    fig, ax = plt.subplots()

    GT_files=natsorted(os.listdir(Sequence_path+'/GT'))
    Radar_files=natsorted(os.listdir(Sequence_path+'/Radar_Data'))
    for index in range(len(GT_files)):
        GT = np.load(os.path.join(Sequence_path+'/GT/', GT_files[index]))
        Radar = np.load(os.path.join(Sequence_path+'/Radar_Data/', Radar_files[index]))
        # vertical flip:
        Radar = np.flipud(Radar)
        # Clear the previous plot
        ax.clear()
        # Plot the new data
        plt.imshow(Radar, cmap='gray',extent=[-60,60,0,50])
        ax.scatter(GT[:, 0], GT[:, 1],c='r',s=2)

        plt.xlim(-60, 60)
        plt.ylim(0, 50)

        plt.pause(0.1)


def visualize_modified_data(Sequence_path,GT_Output_shape):
    from .Data_Loaders import load_sequence_data
    for sequence_path in Sequence_path:
        #Raw data:
        gt_files = natsorted(os.listdir(os.path.join(sequence_path, 'GT')))
        raw_gt_data = [np.load(os.path.join(sequence_path, 'GT', file)) for file in gt_files]
        radar_data, gt_data = load_sequence_data(sequence_path,GT_Output_shape)
        for radar, gt ,raw_gt in zip(radar_data, gt_data,raw_gt_data):
            plt.clf()
            plt.imshow(radar, cmap='gray',extent=[-60,60,0,50])
            plt.scatter(raw_gt[:, 0], raw_gt[:, 1],c='b',s=2)
            for i in range(GT_Output_shape[0]):
                plt.plot([i*120/(GT_Output_shape[0]+1)-60,(i+1)*120/(GT_Output_shape[0]+1)-60], [gt[i],gt[i]],c='r')
            plt.xlim(-60, 60)
            plt.ylim(0, 50)
            plt.pause(0.1)