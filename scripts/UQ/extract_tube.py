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

parser = argparse.ArgumentParser()
parser.add_argument('-vtu_list')
parser.add_argument('-tube_file')
parser.add_argument('-label_file')
parser.add_argument('-output_fn')
parser.add_argument('-surf_output_dir')
args = parser.parse_args()

VTUS = open(args.vtu_list,'r').readlines()
VTUS = [v.replace('\n','') for v in VTUS]

LABELS = open(args.label_file,'r').readlines()
LABELS = [v.replace('\n','') for v in LABELS]

TUBE_FILE = io.load_json(args.tube_file)
POINTS    = TUBE_FILE['points']
NORMALS   = TUBE_FILE['normals']
RADIUSES  = TUBE_FILE['radiuses']

data = []

for i,vtu_fn in enumerate(VTUS[1:5]):
    if not os.path.exists(vtu_fn): continue

    pd = sv.read_vtu(vtu_fn)

    for p,n,r in zip(POINTS, NORMALS, RADIUSES):
        surf = sv.clip_plane_rad(pd,p,n,r)

        area = sv.vtk_integrate_triangle_surface(surf)
        print(i,area)
    # for j in range(NUM_POINTS):
    #     p = LINE[j]
    #     d = {"model":i,
    #         "point":j,
    #          "x":p[0],
    #          "y":p[1],
    #          "z":p[2]}
    #
    #     for l in LABELS:
    #         tup = v.GetPointData().GetArray(l).GetTuple(j)
    #
    #         for k in range(len(tup)):
    #             d[l+'_'+str(k)] = tup[k]
    #
    #     data.append(d)
#
# df = pandas.DataFrame(data)
# df.to_csv(args.output_fn)

writer = vtk.vtkPolyDataWriter()
for i,d in enumerate(data):
    writer.SetInputData(d)
    writer.SetFileName('/home/marsdenlab/projects/SV/UQ/data/2_vessel_data/tubes/'+str(i)+'.vtu')
    writer.Write()
