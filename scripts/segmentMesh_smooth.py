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

if "REMESH_SIZE" in cfg:
    REMESH_SIZE = cfg['REMESH_SIZE']
else:
    REMESH_SIZE = 0.1

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

################################################################################
# 1. Environment Setup
################################################################################
sys.path.append(SV_PATH)

#chdir needed to find sv shared object files (.so)
os.chdir(SV_BUILD_PATH)

import sv
sv.Solid.SetKernel('PolyData')
solid = sv.Solid.pySolidModel()

#Create model from polydata
MERGE_NAMES = cfg['INTERSECTS']
pds = []

for nc in MERGE_NAMES:
    fn = OUTPUT_DIR+'/'+nc+'.vtp'
    solid.NewObject(nc)
    sv.Repository.ReadVtkPolyData(nc+'_pd', fn)
    solid.SetVtkPolyData(nc+'_pd')

solid.Union("model_0", MERGE_NAMES[0], MERGE_NAMES[1])
for i,m in enumerate(MERGE_NAMES[2:]):
    solid.Union("model_"+str(i+1), "model_"+str(i), m)

solid.GetBoundaryFaces(50)
solid.GetFaceIds()
solid.GetPolyData("model_pd")

#solid.GetPolyData(s3)
print("remeshing")
sv.MeshUtil.Remesh("model_pd", "model_remesh", REMESH_SIZE,REMESH_SIZE)
sv.MeshUtil.Remesh("model_remesh", "model_pd_smooth_0", REMESH_SIZE,REMESH_SIZE)

#Constrain smoothing
s3 = "model_pd_smooth_0"
if "LOCAL_SMOOTH" in cfg:
    points = cfg['LOCAL_SMOOTH']['POINTS']
    radius = cfg['LOCAL_SMOOTH']['RADIUS']
    n = len(points)
    for i,p,r in zip(range(n),points,radius):
        s1 = "model_pd_smooth_{}".format(i)
        s2 = "model_pd_sphere_{}".format(i)
        s3 = "model_pd_smooth_{}".format(i+1)
        print(s1,s2,s3)
        sv.Geom.Set_array_for_local_op_sphere(s1,s2, r, p)
        sv.Geom.Local_constrain_smooth(s2,s3, 10, 0.5)

print("done smoothing")

sv.Repository.WriteVtkPolyData(s3,"ascii",EXTERIOR_FILE)

solid.SetVtkPolyData(s3)
solid.GetBoundaryFaces(50)
solid.GetFaceIds()

#Extract boundary faces
print ("Creating model: \nFaceID found: " + str(solid.GetFaceIds()))

fids = {"faceids":solid.GetFaceIds()}
io.write_json(fids,FACE_FILE)
#Write to file
solid.WriteNative(EXTERIOR_FILE)
