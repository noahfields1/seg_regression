import argparse
import os
import sys
import vtk

sys.path.append(os.path.abspath('..'))

import modules.io as io

parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-output_dir')

args = parser.parse_args()

cfg = io.load_json(args.config)

OUTPUT_DIR = args.output_dir
FACE_FILE = OUTPUT_DIR+'/faceids.json'

CAP_NAMES = cfg["CAP_NAMES"]

fids = io.load_json(FACE_FILE)
NUM_FACES = len(fids['faceids'])
NUM_CAPS = len(cfg['CAP_NAMES'])
NUM_WALLS = NUM_FACES-NUM_CAPS
print(fids, NUM_FACES, NUM_CAPS, NUM_WALLS)

################################################################################
# 1 Set up filepaths
################################################################################
ug_fn = OUTPUT_DIR+"/mesh_ug.vtk"
pd_fn = OUTPUT_DIR+"/mesh_pd.vtk"

wall_ids = list(range(1,NUM_WALLS+1))

print(wall_ids)

cap_ids = list(range(NUM_WALLS+1, NUM_FACES+1))
caps = {}
for cn,ci in zip(CAP_NAMES, cap_ids):
    caps[ci] = cn
print(caps)

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
