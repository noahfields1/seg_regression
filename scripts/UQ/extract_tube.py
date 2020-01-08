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
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-tube_file')
parser.add_argument('-output_fn')
args = parser.parse_args()

cfg = io.load_json(args.config)

DIR          = cfg['dir']
MESH_LABELS  = cfg['mesh_labels']
GENS         = cfg['generations']
NUM_MODELS   = cfg['num_models']
SIM_NAME     = cfg['sim_name']
RESULTS_FILE = cfg['results_file']

LABS = cfg['labels']
LABELS = []
for i in range(cfg['start'], cfg['end'], cfg['incr']):
    for l in LABS:
        new_l = str(i)
        if len(new_l) == 1:
            new_l = "0000"+new_l
        if len(new_l) == 2:
            new_l = "000"+new_l
        if len(new_l) == 3:
            new_l = "00"+new_l
        if len(new_l) == 4:
            new_l = "0"+new_l
        LABELS.append(l+'_'+new_l)

TUBE_FILE = io.load_json(args.tube_file)
POINTS    = TUBE_FILE['points']
NORMALS   = TUBE_FILE['normals']
RADIUSES  = TUBE_FILE['radiuses']
N         = range(len(POINTS))

data = []

for gen in GENS:
    for mesh in MESH_LABELS:
        for nm in range(2):
            vtu_fn = DIR+'/'+str(gen)+'/'+mesh+'/'+str(nm)+'/'+SIM_NAME+'/'+RESULTS_FILE
            print(vtu_fn)
            if not os.path.exists(vtu_fn): continue

            pd = sv.read_vtu(vtu_fn)

            for j,p,n,r in zip(N, POINTS, NORMALS, RADIUSES):
                surf = sv.clip_plane_rad(pd,p,n,r)

                #try:
                area = sv.vtk_integrate_pd_area(surf)
                length = sv.vtk_integrate_pd_boundary_length(surf)

                d = {"generation":gen,
                    "mesh":mesh,
                    "model":nm,
                    "point":j,
                     "x":p[0], "y":p[1], "z":p[2],
                     "nx":n[0],"ny":n[1],"nz":n[2],
                     "radius_actual":np.sqrt(area/np.pi),
                     "radius_supplied":r,
                     "area":area,
                     "length":length
                     }

                for l in LABELS:
                    print(l)
                    ints    = sv.vtk_integrate_pd_volume(surf, l)
                    ints_bd = sv.vtk_integrate_pd_boundary(surf, l)

                    for k in range(len(ints)):
                        d[l+'_'+str(k)] = ints[k]*1.0/area
                        d[l+'_'+str(k)+'_boundary'] = ints_bd[k]*1.0/length

                data.append(d)
                #except:
                    #print("failed vtu {} point {}".format(i,j))

df = pandas.DataFrame(data)
df.to_csv(args.output_fn)
