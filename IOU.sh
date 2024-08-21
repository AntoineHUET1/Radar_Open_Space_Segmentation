#!/bin/bash

# Base command for testing
BASE_CMD="python main.py --mode test --label test --config_path"

for RANGE in {25..40..5}
do
  # Test with the current radar range
  CONFIG_PATH="Results/Radar/${RANGE}m/Experience_2/config.py"
  echo "Testing with radar_range=${RANGE}"
  $BASE_CMD $CONFIG_PATH
done
