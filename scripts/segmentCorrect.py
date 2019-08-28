import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

from modules import vascular_data as sv
from modules import io
import numpy as np
from tqdm import tqdm

from sv_wrapper import SVWrapper

parser = argparse.ArgumentParser()
parser.add_argument('-input_dir')
parser.add_argument('--deletions')
parser.add_argument('--corrections')

args = parser.parse_args()

groups_files = [f in os.listdir(args.input_dir) if '.json' in f]
if len(groups_files)==0:
    raise RuntimeError("No json groups found")

names = [g.split('/')[-1].replace('.json','') for g in groups_files]
data = [io.load_json(g) for g in groups_files]
