#!/bin/bash

echo "Changing directories"
cd /home/gdmaher/seg_regression/scripts/UQ

echo "Running python with ${1}"
python3.6 run_coronary_cluster.py -vtu ${1}
