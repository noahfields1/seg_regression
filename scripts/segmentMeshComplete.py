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
FACE_FILE = OUTPUT_DIR+'/faceids.json'
FACE_IDS = io.load_json(FACE_FILE)

CAP_NAMES = cfg["CAP_NAMES"]

################################################################################
# 1 Set up filepaths
################################################################################
ug_fn = OUTPUT_DIR+"/mesh_ug.vtk"
pd_fn = OUTPUT_DIR+"/mesh_pd.vtk"

OUT_DIR  = OUTPUT_DIR+"/mesh-complete"
SURF_DIR = OUT_DIR+"/mesh-surfaces"

ug_out_fn = OUT_DIR+"/mesh.vtu"
pd_out_fn = OUT_DIR+"/exterior.vtp"

walls_combined_fn = OUT_DIR+"/walls_combined.vtp"

try:
    os.mkdir(OUT_DIR)
    os.mkdir(SURF_DIR)
except:
    print("mesh complete folders already exist")
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
#unstructured grid
ug = read_vtk(ug_fn)

ugwriter = vtk.vtkXMLUnstructuredGridWriter()
ugwriter.SetCompressorTypeToZLib()
ugwriter.EncodeAppendedDataOff()
ugwriter.SetInputDataObject(ug)
ugwriter.SetFileName(ug_out_fn)
ugwriter.Write()

#polydata
pd = read_vtk(pd_fn)

pdwriter = vtk.vtkXMLPolyDataWriter()
pdwriter.SetCompressorTypeToZLib()
pdwriter.EncodeAppendedDataOff()
pdwriter.SetInputDataObject(pd)
pdwriter.SetFileName(pd_out_fn)
pdwriter.Write()

################################################################################
# 3.5 Compute wall and cap ids
################################################################################
normals  = cfg['NORMALS']
norm_tol = cfg['NORMAL_TOLERANCE']

pd = sv.vtk_pd_compute_normals(pd)

pd_norms = pd.GetPointData().GetArray("Normals")
pd_ids   = pd.GetPointData().GetArray("ModelFaceID")
N        = pd.GetNumberOfPoints()

cap_ids  = {}
wall_ids = []

for name,n in normals.items():
    for i in range(N):
        no = pd_norms.GetTuple(i)

        ip = np.abs( no[0]*n[0] + no[1]*n[1] + no[2]*n[2] )-1
        if ip < norm_tol:
            id = pd_ids.GetTuple(i)[0]
            cap_ids[name] = id
            break

import pdb; pdb.set_trace()
################################################################################
# 4 Output everything
################################################################################
#walls combined
appender = vtk.vtkAppendPolyData()
appender.UserManagedInputsOff()
for id in wall_ids:
    w_pd = thresholdPolyData(pd, "ModelFaceID", (id,id))
    appender.AddInputData(w_pd)

appender.Update()

cleaner =vtk.vtkCleanPolyData()
cleaner.PointMergingOn()
cleaner.PieceInvariantOff()
cleaner.SetInputDataObject(appender.GetOutput())
cleaner.Update()

pdwriter.SetFileName(walls_combined_fn)
pdwriter.SetInputDataObject(cleaner.GetOutput())
pdwriter.Write()

#caps
for k,v in caps.items():

    c_pd = thresholdPolyData(pd, "ModelFaceID", (k,k))
    fn   = SURF_DIR + '/'+v+'.vtp'

    pdwriter.SetFileName(fn)
    pdwriter.SetInputDataObject(c_pd)
    pdwriter.Write()
