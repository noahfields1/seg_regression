import sys
import os
sys.path.append(os.path.abspath('..'))

import vtk
import modules.io as io
import modules.vascular_data as sv

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-vtu')
parser.add_argument('-coord_file')
parser.add_argument('-labels_file')
parser.add_argument('-output_dir')
parser.add_argument('-model_id')

args     = parser.parse_args()

vtu_fn   = args.vtu

coord_fn     = args.coord_file
coord_lines  = open(coord_fn,'r').readlines()
coord_lines  = [s.replace('\n','') for s in coord_lines]
coord_lines  = [s.split(',') for s in coord_lines]

coord_ids     = [c[0] for c in coord_lines]
coord_points  = [[float(c[1]),float(c[2]),float(c[3])] for c in coord_lines]

labels_fn  = args.labels_file
labels     = open(labels_fn,'r').readlines()
labels     = [v.replace('\n','') for v in labels]

output_fn  = coord_fn.split('/')[-1].replace('.txt', '.csv')
output_dir = args.output_dir

output_fn = output_dir+'/'+output_fn

try:
    os.mkdir(output_dir)
except:
    print(output_dir, " already exists")

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(vtu_fn)
reader.Update()

polyd = reader.GetOutput()

# #########################################
# #interpolate
# #########################################
temp_data = []
for i in range(len(coord_lines)):
    p = coord_points[i]
    temp_data.append( {"index":i,
                        "coord_id":coord_ids[i],
                        "model_id":args.model_id,
                        "x":p[0],
                        "y":p[1],
                        "z":p[2]} )

for l in labels:
    data   = polyd.GetPointData().GetArray(l)
    tup_dim = len(data.GetTuple(0))

    for tn in range(tup_dim):
        new_label = l+'_'+str(tn)
        interp = sv.get_interp(polyd, l, tn)

        for i,coord in enumerate(coord_lines):

            p = coord_points[i]
            v = interp(p)[0]

            temp_data[i][new_label] = v

df = pd.DataFrame(temp_data)

df.to_csv(output_fn)
