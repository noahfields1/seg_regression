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

def thresholdPolyData(poly, attr, threshold):
    """
    Get the polydata after thresholding based on the input attribute
    Args:
        poly: vtk PolyData to apply threshold
        atrr: attribute of the cell array
        threshold: (min, max)
    Returns:
        output: vtk PolyData after thresholding
    """
    surface_thresh = vtk.vtkThreshold()
    surface_thresh.SetInputData(poly)
    surface_thresh.SetInputArrayToProcess(0,0,0,1,attr)
    surface_thresh.ThresholdBetween(*threshold)
    surface_thresh.Update()
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(surface_thresh.GetOutput())
    surf_filter.Update()
    return surf_filter.GetOutput()

################################################################################
# 3 Write files
################################################################################
#polydata
pd = sv.vtk_read_native_polydata(EXTERIOR_FILE)

################################################################################
# 3.5 Compute wall and cap ids
################################################################################
eps = 0.1
cap_locs = cfg['CAP_LOCS']

pd_ids   = pd.GetCellData().GetArray("ModelFaceID")
N        = pd.GetNumberOfCells()

cap_ids  = {}

ids = list(set([pd_ids.GetTuple(i)[0] for i in range(N)]))
centroids = []
for id in ids:
    pd_th = thresholdPolyData(pd,"ModelFaceID",(id,id))
    n     = pd_th.GetNumberOfCells()
    centroids.append([0.0,0.0,0.0])
    for i in range(n):
        cell = pd_th.GetCell(i)
        points = cell.GetPoints()

        coo = [points.GetTuple(j) for j in range(3)]

        centroids[-1][0] += (coo[0][0]+coo[1][0]+coo[2][0])*1.0/3
        centroids[-1][1] += (coo[0][1]+coo[1][1]+coo[2][1])*1.0/3
        centroids[-1][2] += (coo[0][2]+coo[1][2]+coo[2][2])*1.0/3


import pdb; pdb.set_trace()

for name,p in cap_locs.items():
    id,coord,weights = sv.vtkPdFindCellId(pd,p)

    model_face_id = pd_ids.GetTuple(id)[0]
    print(name,p,model_face_id)
    cap_ids[name] = model_face_id

wall_ids = list(set([pd_ids.GetTuple(i)[0] for i in range(N)]))
wall_ids = [s for s in wall_ids if not any([s == v for k,v in cap_ids.items()])]

io.write_json(cap_ids,OUTPUT_DIR+'/cap_ids.json')
io.write_json(wall_ids,OUTPUT_DIR+'/wall_ids.json')
