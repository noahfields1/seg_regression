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
parser.add_argument('-vtu')
parser.add_argument('-config')
parser.add_argument('-output_fn')
args = parser.parse_args()

cfg = io.load_json(args.config)

START = cfg['START']
STOP  = cfg['STOP']
STEP  = cfg['STEP']
p     = cfg['POINT']
n     = cfg['NORMAL']
r     = cfg['RADIUS']

quant = "pressure"

LABS = []

for i in range(START,STOP,STEP):
    new_l = str(i)
    if len(new_l) == 1:
        new_l = "0000"+new_l
    if len(new_l) == 2:
        new_l = "000"+new_l
    if len(new_l) == 3:
        new_l = "00"+new_l
    if len(new_l) == 4:
        new_l = "0"+new_l
    new_l = quant+'_'+new_l
    LABS.append(new_l)

pd = sv.read_vtu(args.vtu)
data = []
for l in LABS:
    print(l)
    surf = sv.clip_plane_rad(pd,p,n,r)

    area = sv.vtk_integrate_pd_area(surf)
    length = sv.vtk_integrate_pd_boundary_length(surf)

    d = {    "label":quant,
            "time":new_l,
             "x":p[0], "y":p[1], "z":p[2],
             "nx":n[0],"ny":n[1],"nz":n[2],
             "radius_actual":np.sqrt(area/np.pi),
             "radius_supplied":r,
             "area":area,
             "length":length
             }


    ints    = sv.vtk_integrate_pd_volume(surf, l)
    ints_bd = sv.vtk_integrate_pd_boundary(surf, l)

    for k in range(len(ints)):
        d[quant+'_'+str(k)] = ints[k]*1.0/area
        d[quant+'_'+str(k)+'_boundary'] = ints_bd[k]*1.0/length

    data.append(d)

df = pandas.DataFrame(data)
df.to_csv(args.output_fn)
