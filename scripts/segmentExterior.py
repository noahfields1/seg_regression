import argparse
import os
import sys

sys.path.append(os.path.abspath('..'))

import modules.io as io

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

if "MODES" in cfg:
    NUM_MODES = cfg['MODES']
else:
    NUM_MODES = 5

if "FFT" in cfg:
    USE_FFT = cfg['FFT']
else:
    USE_FFT = 0

EXTERIOR_FILE = OUTPUT_DIR+'/exterior.vtp'
UG_FILE = OUTPUT_DIR+'/mesh_ug.vtk'
PD_FILE = OUTPUT_DIR+'/mesh_pd.vtk'
FACE_FILE = OUTPUT_DIR+'/faceids.json'

FILES = os.listdir(INPUT_DIR)
FILES = [INPUT_DIR+'/'+f for f in FILES]

#NOTE: these should all sort into the same order, i.e.
#paths and group files should use the same names
PATH_FILES    = sorted(cfg['PATH_FILES'])
NAMES         = sorted(cfg['NAMES'])
GROUPS_FILES  = [INPUT_DIR+'/'+n+'_corrected.json' for n in NAMES]
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
sys.path.append(SV_PATH)

#chdir needed to find sv shared object files (.so)
os.chdir(SV_BUILD_PATH)

import sv
################################################################################
# 4. Cap lofted vessels and merge into solid model
################################################################################
#Set solid kernel
sv.Solid.SetKernel('PolyData')
solid = sv.Solid.pySolidModel()

#Create model from polydata
MERGE_NAMES = cfg['INTERSECTS']

print(NAMES_LOFT)
print(NAMES_CAP)
print(MERGE_NAMES)

for nc in MERGE_NAMES:
    fn = OUTPUT_DIR+'/'+nc+'.vtp'
    print(nc)
    print(fn)
    solid.NewObject(nc)
    sv.Repository.ReadVtkPolyData(nc, fn)
    solid.SetVtkPolyData(nc)

l = sv.Repository.List()

print(MERGE_NAMES[0]+'_new', MERGE_NAMES[1]+'_new')
solid.Union("model_0", MERGE_NAMES[0]+'_new', MERGE_NAMES[1]+'_new')
for i,m in enumerate(MERGE_NAMES[2:]):
    print("union {} {}".format(i,m))
    solid.Union("model_"+str(i+1), "model_"+str(i), m+'_new')


#sv.Geom.Clean("model_pd","model_pd_clean")
#solid.SetVtkPolyData("model_pd_clean")
solid.GetBoundaryFaces(50)
solid.GetFaceIds()
solid.GetPolyData("model_pd")

#sv.Geom.Set_array_for_local_op_sphere("model_pd","model_pd_1", 3.0, [0.28,15.74,-43.24])
#sv.Geom.Local_constrain_smooth("model_pd_1","model_pd_2", 10, 0.2)
print(EXTERIOR_FILE)
sv.Repository.WriteVtkPolyData("model_pd","ascii",EXTERIOR_FILE)

import pdb; pdb.set_trace()
