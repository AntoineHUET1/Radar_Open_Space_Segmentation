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
# Dataset

Leddar PixSet Dataset is a publicly available dataset containing approximately 29k frames from 97 sequences recorded in high-density urban areas, using a set of various sensors (cameras, LiDARs, radar, IMU, etc.).

### Download Ground Truth Data

**Direct Download** from this google drive link: [Ground Truth Data](https://drive.google.com/file/d/1J9)

or 

**Generate Locally** using the following steps:
1. Download the [Leddar PixSet Dataset](https://dataset.leddartech.com/) from the official website.
2. Extract the dataset in the root directory of the project in data folder.
3. Run the following script to generate ground truth data:
```bash
GenerateGroundTruthData.py
```

### Download Radar Data

**Direct Download** from this Google Drive link: [Radar Data ](https://drive.google.com/file/d/1J9)


## Data Preparation:

Ounce ground truth and radar data are downloaded and extracted.Create link to data directory.
```bash
cd $PROJECTROOT
mkdir -p data
ln -s $RADARDATA data/RadarData
ln -s $GROUNDTRUTH data/GroundTruthData
```



