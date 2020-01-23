import argparse
import os
import sys
import vtk

sys.path.append(os.path.abspath('..'))
import numpy as np
import modules.io as io
import modules.vascular_data as sv

parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-output_dir')

args = parser.parse_args()

cfg = io.load_json(args.config)

OUTPUT_DIR = args.output_dir

CAP_NAMES = cfg["CAP_NAMES"]

EXTERIOR_FILE = OUTPUT_DIR+'/exterior.vtp'

################################################################################
# 2 define some functions
################################################################################
def read_vtk(fn):
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(fn)
    reader.Update()
    return reader.GetOutput()

################################################################################
# 3 Write files
################################################################################
#polydata
pd = sv.vtk_read_native_polydata(EXTERIOR_FILE)

################################################################################
# 3.5 Compute wall and cap ids
################################################################################
cap_locs = cfg['CAP_LOCS']

pd_ids   = pd.GetCellData().GetArray("ModelFaceID")
N        = pd.GetNumberOfCells()

cap_ids  = {}

for name,p in cap_locs.items():
    id,coord,weights = sv.vtkPdFindCellId(pd,p)

    model_face_id = pd_ids.GetTuple(id)[0]
    cap_ids[name] = model_face_id

wall_ids = list(set([pd_ids.GetTuple(i)[0] for i in range(N)]))
wall_ids = [s for s in wall_ids if not any([s == v for k,v in cap_ids.items()])]

io.write_json(cap_ids,OUTPUT_DIR+'/cap_ids.json')
io.write_json(wall_ids,OUTPUT_DIR+'/wall_ids.json')
