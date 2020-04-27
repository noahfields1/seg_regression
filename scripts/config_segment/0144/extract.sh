#!/bin/bash

echo "Changing directories"
cd seg_regression

echo "Running python with ${1}"
python3.6 train.py ${1} TRAIN
