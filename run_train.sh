#!/bin/bash

for mode in 1 0
do
  for range in {5..50..5}
  do
      python main.py --mode train --radar_range $range --GT_mode $mode
  done
done

