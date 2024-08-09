# ROSS: A Deep Convolutional Network for Radar Open Space Segmentation



ROSS is a deep convolutional neural network designed to process raw radar data for open space segmentation in autonomous driving. It efficiently detects the boundary of drivable space using a simplified model based on the LeNet architecture.

![Radar_open_space_segmentation.jpg](Images/Radar_open_space_segmentation.png)

# Table of Contents

1. [Installation](#Installation) 
2. [Dataset](#Dataset)
3. [Data Preparation](#Data-Preparation)

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

### [Optional] Manually generate the dataset

1. Download the [Leddar PixSet Dataset](https://dataset.leddartech.com/) from the official website.
2. Extract the dataset in the root directory of the project in data folder.
    ```bash
   cd $PROJECTROOT
    mkdir -p data
    ln -s $PIXETDATASET data/Pixet_Dataset
    ```
3. Run the following script to generate ground truth data:
    ```bash
    GenerateGroundTruthData.py
    ```

## Data Preparation:

Once the dataset is downloaded and extracted, create a symbolic link to the data directory.
```bash
cd $PROJECTROOT
mkdir -p data
ln -s $DATASET data/ROSS_Dataset
```




