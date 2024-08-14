import os
import tensorflow as tf
from ROSS.Utils import genrerat_Graph, Generate_Data
from ROSS.Model import build_ROSS_32_50, Train_model, Metric_MAE, CustomLoss
import warnings
import ROSS.cfg.ROSS_Config as cfg
import json


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
os.chdir(script_dir)

# Filter out specific TensorFlow warnings
warnings.filterwarnings("ignore", message="TF-TRT Warning")

default_config_path = '/ROSS/cfg/ROSS_Config.py'

config_Path='/home/watercooledmt/PycharmProjects/Radar_Open_Space_Segmentation/Results/Radar/25m/Experience_1/config.py'

Save_path = './Results'
if not os.path.exists(Save_path):
    os.makedirs(Save_path)

# ==================== Set Up Data ====================

# List files in the ROSS_Dataset directory
Sequence_List = os.listdir(cfg.Data_path)

# Remove bad sequences from list:
if cfg.Remove_bad_sequences:
    Sequence_List = [seq for seq in Sequence_List if seq not in cfg.Bad_sequences]

# Test, Train, Validation split:
Number_of_files = len(Sequence_List)
Val_files = int(Number_of_files * cfg.Val_ratio)
Test_files = int(Number_of_files * cfg.Test_ratio)
Train_files = Number_of_files - Val_files - Test_files

Train_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[:Train_files]]
Val_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[Train_files:Train_files + Val_files]]
Test_sequence_paths = [cfg.Data_path + seq for seq in Sequence_List[Train_files + Val_files:]]

# Radar_Range:
if cfg.Radar_Range <= 25:
    input_shape = (128, 256, 1)

# ==================== Model ====================

acc_metric = Metric_MAE(cfg)
loss = CustomLoss()

# ==================== Callbacks ====================

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.1,
        patience=cfg.patience,
        verbose=0,
        mode="auto",
        min_lr=0.000001,
    )
]


def train_model(cfg):

    if not config_Path:
        run_dir = Save_path + f"/Radar/{cfg.Radar_Range}m/"

        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # Create Experience i directory inside run_dir
        Existing_Experiences = os.listdir(run_dir)
        Experience_number = len(Existing_Experiences) + 1
        run_dir = run_dir + f'/Experience_{Experience_number}'
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        # Save the configuration file in the experience directory
        os.system(f"cp {script_dir+default_config_path} {run_dir}/config.py")
    else:
        run_dir = os.path.dirname(config_Path)


    # Generate data
    train_dataloader, train_dataloader_length, val_dataloader, val_dataloader_length, test_dataloader, test_dataloader_length = Generate_Data(cfg, Train_sequence_paths, Val_sequence_paths, Test_sequence_paths)

    model = build_ROSS_32_50(cfg)

    model.summary()

    print('------------------------------')
    print('Training model')
    print('------------------------------')
    Train_model(cfg, model, acc_metric, loss, train_dataloader, train_dataloader_length, val_dataloader,val_dataloader_length,
                                test_dataloader, test_dataloader_length, run_dir)

    print('\n------------------------------')
    print('Generating Graphs')
    print('------------------------------')
    # Best model path
    checkpoint_path=run_dir+"/best_model_weights.weights.h5"

    # Generate graph:
    genrerat_Graph(checkpoint_path, test_dataloader, cfg, label='test', Save_fig=True)
    genrerat_Graph(checkpoint_path, train_dataloader, cfg, label='train', Save_fig=True)
    genrerat_Graph(checkpoint_path, val_dataloader, cfg, label='val', Save_fig=True)


train_model(cfg)
