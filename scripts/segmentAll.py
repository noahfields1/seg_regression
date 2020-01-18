import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
from tqdm import tqdm
from shutil import copyfile, rmtree

import modules.io as io

parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-output_dir')
args = parser.parse_args()

cfg = io.load_json(args.config)

IMAGE = cfg["IMAGE"]
PATHS = cfg["PATH_FILES"]
NN_CONFIG = cfg["NN_CONFIG"]
if "MODES" in cfg:
    MODES = cfg['MODES']
else:
    MODES = 5

OUTPUT = os.path.abspath(args.output_dir)

try:
    rmtree(OUTPUT)
except:
    pass

os.mkdir(OUTPUT)

if not os.path.isfile(IMAGE) or not os.path.isdir(OUTPUT):
    raise RuntimeError('error occurred {} {} {}'.format(IMAGE,PATHS,OUTPUT))

else:
    for p in PATHS:
        os.system('python segmentCompute.py\
             -image_file {} -points_file {} -output_dir {}\
              -config {} -modes {}'.format(IMAGE, p, OUTPUT, NN_CONFIG, MODES))
