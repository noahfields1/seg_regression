import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

import numpy as np
from tqdm import tqdm
from shutil import copyfile, rmtree

parser = argparse.ArgumentParser()
parser.add_argument('-input_dir')
parser.add_argument('-output_dir')

parser.add_argument('-config')
args = parser.parse_args()

DIR = os.path.abspath(args.input_dir)
IMAGE = DIR+'/image.mha'
PATHS = DIR+'/paths'
OUTPUT = os.path.abspath(args.output_dir)

try:
    rmtree(OUTPUT)
except:
    pass

os.mkdir(OUTPUT)

path_files = os.listdir(PATHS)
path_files = [PATHS+'/'+p for p in path_files]

copyfile(DIR+'/group_contents.tcl',OUTPUT+'/group_contents.tcl')

if not os.path.isfile(IMAGE) or not os.path.isdir(PATHS) or not os.path.isdir(DIR):
    raise RuntimeError('error occurred {} {} {}'.format(IMAGE,PATHS,DIR))

else:
    for p in path_files:
        os.system('python computeSegmentations.py\
             -image_file {} -points_file {} -output_dir {}\
              -config {}'.format(IMAGE, p, OUTPUT, args.config))
