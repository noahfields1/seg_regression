import argparse
import os
import sys
import time
sys.path.append(os.path.abspath('../..'))
import pandas
import json
import modules.io as io
import modules.vascular_data as sv
import vtk
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-tube_file')
parser.add_argument('-output_fn')
args = parser.parse_args()

cfg = io.load_json(args.config)

DIR          = cfg['dir']
MESH_LABELS  = cfg['mesh_labels']
GENS         = cfg['generations']
NUM_MODELS   = cfg['num_models']
SIM_NAME     = cfg['sim_name']
RESULTS_FILE = cfg['results_file']

TUBE_FILE = io.load_json(args.tube_file)
POINTS    = TUBE_FILE['points']
NORMALS   = TUBE_FILE['normals']
RADIUSES  = TUBE_FILE['radiuses']
N         = range(len(POINTS))

data = []

for gen in GENS:
    for mesh in MESH_LABELS:
        for nm in range(NUM_MODELS):
            vtu_fn = DIR+'/'+str(gen)+'/'+mesh+'/'+str(nm)+'/'+SIM_NAME+'/'+RESULTS_FILE
            print(vtu_fn)
            if not os.path.exists(vtu_fn): continue

            os.system('python extract_tube_single.py ')
