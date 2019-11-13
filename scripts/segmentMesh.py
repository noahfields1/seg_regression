import argparse
import os
import sys

sys.path.append(os.path.abspath('..'))

import modules.io as io

parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-output_dir')

args = parser.parse_args()

cfg = io.load_json(args.config)

OUTPUT_DIR = args.output_dir
SV_PATH       = cfg['SV_PATH']
SV_BUILD_PATH = cfg['SV_BUILD_PATH']

INTERVAL = cfg["INTERVAL"]

EDGE_SIZE = cfg["EDGE_SIZE"]

EXTERIOR_FILE = OUTPUT_DIR+'/exterior.vtp'
UG_FILE = OUTPUT_DIR+'/mesh_ug.vtk'
PD_FILE = OUTPUT_DIR+'/mesh_pd.vtk'
FACE_FILE = OUTPUT_DIR+'/faceids.json'

FILES = os.listdir(OUTPUT_DIR)
FILES = [OUTPUT_DIR+'/'+f for f in FILES]

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
sys.path.append(SV_PATH)

#chdir needed to find sv shared object files (.so)
os.chdir(SV_BUILD_PATH)

import sv

################################################################################
# 2. Create Paths and Groups
################################################################################
PD_NAMES = {}
for name,path,group in zip(NAMES, PATHS, GROUPS):
    print(name)

    #path
    sv_path = sv.Path.pyPath()
    sv_path.NewObject(name)

    for point in path:
        p = list(point)
        sv_path.AddPoint(p[1:4])

    sv_path.CreatePath()
    print(sv_path.GetPathPtsNum())

    #contour
    kernel = "SplinePolygon"
    sv.Contour.SetContourKernel(kernel)

    path_pts = sv_path.GetPathPosPts()

    point_per_id = (sv_path.GetPathPtsNum()-1)/(len(path)-1)
    point_per_id = int(point_per_id)

    print(point_per_id)

    PD_NAMES[name] = []

    for i,t in enumerate(group.items()):
        k,v = t
        k = int(k)
        group_pos = int(k/INTERVAL)
        pos = int(group_pos*point_per_id)

        contour = sv.Contour.pyContour()

        contour_name = name+'_'+str(pos)

        contour.NewObject(contour_name, name, pos)

        ctrlPts = v
        contour.SetCtrlPts(v)
        contour.Create()

        pd_name = contour_name+'_pd'
        PD_NAMES[name].append(pd_name)
        contour.GetPolyData(pd_name)

################################################################################
# 3. Loft Segmentations
################################################################################
numOutPtsInSegs = 60
numOutPtsAlongLength = 100
numPtsInLinearSampleAlongLength = 240
numLinearPtsAlongLength = 120
numModes = 20
useFFT = 1
useLinearSampleAlongLength = 1

sample_names = {}
aligned_names = {}
for name in NAMES:
    sample_names[name] = []
    aligned_names[name] = []

    for pd_name in PD_NAMES[name]:
        s_name = pd_name+'_s'
        sv.Geom.SampleLoop(pd_name,numOutPtsInSegs,s_name)
        sample_names[name].append(s_name)

    #align contours
    n1 = sample_names[name][0]
    aligned_names[name].append(n1)
    for i in range(1,len(sample_names[name])):
        n2 = sample_names[name][i]
        na = n2+'_aligned'
        sv.Geom.AlignProfile(n1,n2,na,0)
        aligned_names[name].append(na)

        n1 = na

    #loft contours
    srcList = aligned_names[name]
    dstName = name+'_loft'

    sv.Geom.LoftSolid(srcList,dstName,numOutPtsInSegs,
                   numOutPtsAlongLength,numLinearPtsAlongLength,
                   numModes,useFFT,useLinearSampleAlongLength)

################################################################################
# 4. Cap lofted vessels and merge into solid model
################################################################################
#Set solid kernel
for nl,nc in zip(NAMES_LOFT, NAMES_CAP):
    sv.VMTKUtils.Cap_with_ids(nl,nc,0,0)

if len(NAMES_CAP) == 1:
    s = NAMES_CAP[0]
else:
    s = "model_loft_cap_1"
    sv.Geom.Intersect(NAMES_CAP[0], NAMES_CAP[1], s)
    if len(NAMES_CAP)>2:
        for i in range(2, len(NAMES_CAP)):
            ss = "model_loft_"+str(i)
            sv.Geom.Intersect(NAMES_CAP[i], s, ss)
            s = ss

sv.Repository.WriteVtkPolyData(nc, 'ascii', OUTPUT_DIR+'/test.vtk')

sv.Solid.SetKernel('PolyData')
#Create model from polydata
solid = sv.Solid.pySolidModel()
solid.NewObject('model')
solid.SetVtkPolyData(s)
solid.GetBoundaryFaces(90)
solid.GetPolyData("model_pd")

sv.MeshUtil.Remesh("model_pd", "model_remesh", 0.1,0.1)
sv.MeshUtil.Remesh("model_remesh", "model_remesh_2", 0.1,0.1)

solid.SetVtkPolyData("model_remesh")

#Extract boundary faces
print ("Creating model: \nFaceID found: " + str(solid.GetFaceIds()))

fids = {"faceids":solid.GetFaceIds()}
io.write_json(fids,FACE_FILE)
#Write to file
solid.WriteNative(EXTERIOR_FILE)

################################################################################
# 5. Create vtk meshes
################################################################################
#Set mesh kernel
sv.MeshObject.SetKernel('TetGen')

#Create mesh object
msh = sv.MeshObject.pyMeshObject()
msh.NewObject('mesh')

#Load Model
solidFn = EXTERIOR_FILE
msh.LoadModel(solidFn)

#Create new mesha
msh.NewMesh()
msh.SetMeshOptions('SurfaceMeshFlag',[1])
msh.SetMeshOptions('VolumeMeshFlag',[1])
msh.SetMeshOptions('GlobalEdgeSize',[EDGE_SIZE])
msh.SetMeshOptions('MeshWallFirst',[1])
msh.SetMeshOptions('Optimization', [3])
msh.SetMeshOptions('QualityRatio', [1.4])
msh.GenerateMesh()

# #Save mesh to file
mesh_ug_name = 'mesh_ug'
mesh_pd_name = 'mesh_pd'

msh.GetUnstructuredGrid(mesh_ug_name)
msh.GetPolyData(mesh_pd_name)

sv.Repository.WriteVtkUnstructuredGrid(mesh_ug_name,"ascii",UG_FILE)
sv.Repository.WriteVtkPolyData(mesh_pd_name,"ascii",PD_FILE)
