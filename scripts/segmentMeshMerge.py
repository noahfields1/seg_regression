import argparse
import os
import sys

sys.path.append(os.path.abspath('..'))

import modules.io as io
import modules.vascular_data as sv

parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-input_dir')
parser.add_argument('-output_dir')
parser.add_argument('-edge_size', type=float)

args = parser.parse_args()

cfg = io.load_json(args.config)

INPUT_DIR     = args.input_dir
OUTPUT_DIR    = args.output_dir
SV_PATH       = cfg['SV_PATH']
SV_BUILD_PATH = cfg['SV_BUILD_PATH']

INTERVAL = cfg["INTERVAL"]

EDGE_SIZE = args.edge_size

EXTERIOR_FILE = OUTPUT_DIR+'/exterior.vtp'
UG_FILE = OUTPUT_DIR+'/mesh_ug.vtk'
PD_FILE = OUTPUT_DIR+'/mesh_pd.vtk'
FACE_FILE = OUTPUT_DIR+'/faceids.json'

FILES = os.listdir(INPUT_DIR)
FILES = [INPUT_DIR+'/'+f for f in FILES]

#NOTE: these should all sort into the same order, i.e.
#paths and group files should use the same names
GROUPS_FILES  = sorted([f for f in FILES if '_corrected.json' in f])
PATH_FILES    = sorted(cfg['PATH_FILES'])
NAMES         = sorted(cfg['NAMES'])
NAMES_LOFT    = [f+'_loft' for f in NAMES]
NAMES_CAP     = [f+'_cap' for f in NAMES]

GROUPS = [io.load_json(f) for f in GROUPS_FILES]
PATHS  = [io.parsePathPointsFile(f) for f in PATH_FILES]

print(NAMES)
print(GROUPS_FILES)
print(PATH_FILES)
################################################################################
# 1. Environment Setup
################################################################################
MERGE_NAMES = cfg['INTERSECTS']
pds = []

for nc in MERGE_NAMES:
    fn = OUTPUT_DIR+'/'+nc+'.vtp'
    pd = sv.vtk_read_polydata(fn)
    pd = sv.vtk_clean_polydata(pd)
    pds.append(pd)

    sv.vtk_write_polydata(pd,fn)
