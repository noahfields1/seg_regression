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

# # ################################################################################
# # # 5. Create vtk meshes
# # ################################################################################
# # # #Set mesh kernel

CAP_IDS = io.load_json(OUTPUT_DIR+'/cap_ids.json')
CAP_IDS = [v for k,v in CAP_IDS.items()]

sv.MeshObject.SetKernel('TetGen')

#Create mesh object
msh = sv.MeshObject.pyMeshObject()
msh.NewObject('mesh')

#Load Model
msh.LoadModel(EXTERIOR_FILE)

#Create new mesha
msh.NewMesh()
msh.SetMeshOptions('SurfaceMeshFlag',[1])
msh.SetMeshOptions('VolumeMeshFlag',[1])
msh.SetMeshOptions('GlobalEdgeSize',[EDGE_SIZE])
msh.SetMeshOptions('MeshWallFirst',[1])
msh.SetMeshOptions('Optimization', [3])
msh.SetMeshOptions('QualityRatio', [1.4])
msh.SetMeshOptions('UseMMG',[0])
for v in CAP_IDS:
    print("local edge size {}".format(v))
    msh.SetMeshOptions('LocalEdgeSize',[v,EDGE_SIZE*1.0/3])
msh.GenerateMesh()

# #Save mesh to file
mesh_ug_name = 'mesh_ug'
mesh_pd_name = 'mesh_pd'

msh.GetUnstructuredGrid(mesh_ug_name)
msh.GetPolyData(mesh_pd_name)

sv.Repository.WriteVtkUnstructuredGrid(mesh_ug_name,"ascii",UG_FILE)
sv.Repository.WriteVtkPolyData(mesh_pd_name,"ascii",PD_FILE)
