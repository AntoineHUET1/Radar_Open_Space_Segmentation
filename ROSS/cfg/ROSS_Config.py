# config.py

# Train Test Val:
Val_ratio = 0.1
Test_ratio = 0.1

# Input and Output shape:
GT_Output_shape = (32, 2)
input_shape = (256, 256, 1)

# List files in the ROSS_Dataset directory
Data_path = './data/ROSS_Dataset/'

# Remove bad sequences:
Remove_bad_sequences = False
Bad_sequences = [
    '20200730_003948_part44_2995_3195', '20200730_003948_part44_5818_6095',
    '20200730_003948_part44_6875_7500', '20200803_151243_part45_1028_1128',
    '20200803_151243_part45_1260_1524', '20200803_151243_part45_2310_2560',
    '20200803_151243_part45_4780_5005', '20200803_174859_part46_1108_1219',
    '20200803_174859_part46_2761_2861', '20200805_002607_part48_2083_2282'
]

# 50 m or 25 m detection:
Half_length = True
Radar_Range = 25  # 50, 25, 20, 15, 10, 5

if Radar_Range <= 25:
    input_shape = (128, 256, 1)

# Merge t-1,t and t+1 Radar images:
Merge_Radar_images = 1  # 0: No, 1: Yes

# ROSS FOV
FOV = 90  # 0: 120, 1: 90

# Have GT data:
GT_mode = 1  # 0: All Data, 1: Only obstacles in range

# Training parameters:
num_epochs = 100
patience = 10  # Number of epochs to wait for improvement

# Hyperparameters:
HP_BATCH_SIZE = 16 # [32, 64, 128]
HP_DROPOUT = 0.2  # [0.1, 0.2, 0.3, 0.5]
HP_LR = 1e-5  # [1e-3, 1e-4, 1e-5]

# Test sequences:
Easy = ['20200706_202209_part31_2636_2746' , '20200610_185206_part1_5095_5195']
Medium = ['20200616_151155_part9_750_900','20200615_184724_part6_5180_5280']
Hard = ['20200706_162218_part21_790_960','20200622_142945_part19_480_700' , '20200615_184724_part6_5180_5280']
Night = ['20200730_003948_part44_2995_3195','20200730_003948_part44_5818_6095','20200730_003948_part44_6875_7500']
Rain = ['20200803_151243_part45_1028_1128','20200803_151243_part45_2310_2560']

# Cache data to save time:
CACHE_FILE = './data/cache.json'
