#!/bin/bash

# Base command for training
BASE_CMD="python main.py --mode train"

# Loop over radar range from 5 to 50 meters, in steps of 5 meters
for GT_MODE in 1 0
do
  # Loop over GT_mode values
  for RANGE in {5..50..5}
  do
    # Train with the current radar range and GT_mode
    echo "Training with radar_range=${RANGE}m and GT_mode=${GT_MODE}"
    $BASE_CMD --radar_range $RANGE --GT_mode $GT_MODE
  done
done