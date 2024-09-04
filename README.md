# ROSS: A Deep Convolutional Network for Radar Open Space Segmentation



ROSS is a deep convolutional neural network designed to process range angle radar data for open space segmentation in autonomous driving. It efficiently detects the boundary of drivable space .

![Radar_open_space_segmentation.jpg](Images/Radar_open_space_segmentation.png)

# Table of Contents

1. [Installation](#Installation) 
2. [Dataset](#Dataset)
3. [Data Preparation](#Data-Preparation)
4. [Usage](#Usage)
    - [Training](#Training)
    - [Visualize data](#Visualize-data)
    - [Test](#Test)
    - [Inference](#Inference)
5. [Main results](#Main-results)

# Installation
### Clone this repository
```bash
git clone https://github.com/AntoineHUET1/Radar_Open_Space_Segmentation.git
```
To install the package, run the following command::
```bash
pip install -r requirements.txt
```
# Dataset

Leddar PixSet Dataset is a publicly available dataset containing approximately 29k frames from 97 sequences recorded in high-density urban areas, using a set of various sensors (cameras, LiDARs, radar, IMU, etc.).

![Ledartech.png](Images/Ledartech.png) 

### Download Dataset

**Direct Download** from this Google Drive link: [Dataset](https://drive.google.com/file/d/13Pai83qq33uq0tttysR4l-IUvQKHGcix/view?usp=sharing)

### [Optional] Manually generate the dataset  (Not available yet)

1. Download the [Leddar PixSet Dataset](https://dataset.leddartech.com/) from the official website.
2. Extract the dataset in the root directory of the project in data folder.
    ```bash
   cd $PROJECTROOT
    mkdir -p data
    ln -s $PIXETDATASET data/Pixet_Dataset
    ```
3. Run the following script to generate ground truth data:
    ```bash
    # Not available yet
    GenerateGroundTruthData.py
    ```

# Data Preparation:

Once the dataset is downloaded and extracted, create a symbolic link to the data directory.
```bash
cd $PROJECTROOT
mkdir -p data
ln -s $DATASET data/ROSS_Dataset
```

# Usage:

### Training

To train the model, run the following command:
```bash
python main.py --mode train
```
This will run the default training parameters stored in [ROSS_Config.py](ROSS%2Fcfg%2FROSS_Config.py)

To resume a interrupted training session, run the following command:
```bash
python main.py --mode train --config_path $CONFIG_PATH
```
with `$CONFIG_PATH` being the path to the configuration file stored in the experiment [results](Results) folder.



#### Command lines arguments for training pipeline

- `--config_path` (optional, string): Path to the configuration file. If provided, this will override the default configuration used in the project.

- `--radar_range` (optional, integer): Defines the range of the radar detection in meters.

- `--ROSS_FOV` (optional, integer): Sets the Field of View (FOV) for the ROSS system in degrees.

- `--GT_mode` (optional, integer): Defines the mode for ground truth (GT) data handling:
  - `0`: Use all available data for training and evaluation.
  - `1`: Use only data where obstacles are within the radar's range.

- `--num_epochs` (optional, integer): Sets the number of epochs to run during training.

- `--patience` (optional, integer): Specifies the number of epochs to wait before stopping training if no improvement is observed. 

### Visualize data

To visualize data, run the following command:
```bash
python main.py --mode visualize
```

#### Command lines arguments for visualization pipeline

- `--sequence` (optional, string): Name of the sequence to visualize. If not provided, a random sequence from the dataset will be used.
- `--radar` (optional, boolean, default: True): Determines whether to visualize radar data. 
- `--camera` (optional, boolean, default: True): Determines whether to visualize front camera data.
- `--GT` (optional, boolean, default: True): Determines whether to visualize ground truth data.
- `--frame_number` (optional, integer): Specifies the frame number to visualize. If not provided, the entire sequence will be visualized.
- `--FPS` (optional, integer): Sets the frames per second for the visualization.

![Data_Visualisation.png](Images/Visualized_Data.png) 
### Test
To test data and generate graphs, run the following command:
```bash
python main.py --mode test
```
#### Command lines arguments for test pipeline
- `--config_path` (required, string): Path to the configuration file stored in the experiment results folder.
- `--model_path` (required, string): Path to the model file.
- `--label` (optional, string): Path to the output label.
- `--show_graph` (optional, boolean): If set, displays the sequence graph.

![Grpah.png](Images/Graph_V2.png) 

### Inference

To inference data, run the following command:
```bash
python main.py --mode inference
```

#### Command lines arguments for inference pipeline

- `--config_path` (required, string): Path to the configuration file stored in the experiment [results](Results) folder you want to use for inference.
- `--label` (optional, string): Path to the output label.
- `--sequence` (optional, string): Name of the sequence for inference.

# Main results

Training results for different radar ranges are available in the following Google Drive links:

### Mode 1: Use only data where obstacles are within the radar's range

| Distance | Mean absolute error | Median | IOU  | Checkpoint + config                                                                                         |
|----------|---------------------|--------|------|-------------------------------------------------------------------------------------------------------------|
| 5 m      | 0.25 m              | 0.2 m  | 0.94 | [Training results](https://drive.google.com/drive/folders/15WR1xnUD40JvMkl29fUR2pmjs-LqOHiy?usp=sharing)    |
| 10 m     | 0.67 m              | 0.2 m  | 0.91 | [Training results](https://drive.google.com/drive/folders/1dIwhLlzFrszEJO8wTMesm_TOUXmI6Jff?usp=sharing)    |
| 15 m     | 1.29 m              | 0.3 m  | 0.87 | [Training results](https://drive.google.com/drive/folders/1ta8wviUKuF2xOcOhcgXYBeS-fRNvRp4f?usp=sharing)    |
| 20 m     | 1.63 m              | 0.4 m  | 0.87 | [Training results](https://drive.google.com/drive/folders/144mVMFSyxSB8j3E4MxBpNpFd9aDX-55V?usp=sharing)    |
| 25 m     | 2.18 m              | 0.5 m  | 0.85 | [Training results](https://drive.google.com/drive/folders/167ngWHshGN8lOevZy5N9w2y38s5Kcd5L?usp=drive_link) |
| 30 m     | 2.48 m              | 0.6 m  | 0.84 | [Training results](https://drive.google.com/drive/folders/1m1LQl7VuLAcfyBZUsYoGe-PnmD6UGAjE?usp=drive_link) |
| 35 m     | 2.72 m              | 0.7 m  | 0.84 | [Training results](https://drive.google.com/drive/folders/1OVX6p4GI6E4GgyuXgtYZNkUG9r67QLP7?usp=sharing)    |
| 40 m     | 3.04 m              | 0.8 m  | 0.83 | [Training results](https://drive.google.com/drive/folders/1Lyzrv4ky1ykm1otWnDAMV7McxGnV1weK?usp=sharing)    |
| 45 m     | 3.12 m              | 0.9 m  | 0.83 | [Training results](https://drive.google.com/drive/folders/1yaKRx9BYs_jqu_fniBpkamdWD3cNdf9D?usp=drive_link) |
| 50 m     | 4.95 m              | 1 m    | 0.79 | [Training results](https://drive.google.com/drive/folders/17QfnVT3C4lG0N-K5k6uLMNkR-pCHopfu?usp=drive_link) |

### Mode 2: Use all available data for training and evaluation

| Distance | Mean absolute error | Median | IOU  | Checkpoint + config                                                                                         |
|----------|---------------------|--------|------|-------------------------------------------------------------------------------------------------------------|
| 25 m     | 3.16 m              | 1 m    | 0.82 | [Training results](https://drive.google.com/drive/folders/1DX4a7a1lopVGg-jSDEVSstfCq4Wg-lZk?usp=drive_link) |
| 30 m     | 3.65 m              | 1.2 m  | 0.81 | [Training results](https://drive.google.com/drive/folders/1R7vHZuVFn_vKYZOsjDbZts9CiIq7Fazz?usp=drive_link) |
| 35 m     | 3.98 m              | 1.4 m  | 0.81 | [Training results](https://drive.google.com/drive/folders/1H5EMNfzCIOImVDcn1nD21pDJD8h_NtuJ?usp=drive_link) |
| 40 m     | 5.05 m              | 1.6 m  | 0.77 | [Training results](https://drive.google.com/drive/folders/1B_4E6bHgDJAjFZq4vf26I2PDgGOtRfqM?usp=drive_link) |