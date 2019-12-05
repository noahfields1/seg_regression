import sys
import os
sys.path.append(os.path.abspath('..'))

import vtk
import modules.io as io
import modules.vascular_data as sv

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-vtu')
parser.add_argument('-group_file')
parser.add_argument('-labels_file')
parser.add_argument('-output_dir')
parser.add_argument('-model_id')

args     = parser.parse_args()

vtu_fn   = args.vtu

labels_fn  = args.labels_file
labels     = open(labels_fn,'r').readlines()
labels     = [v.replace('\n','') for v in labels]

output_dir = args.output_dir

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(vtu_fn)
reader.Update()

polyd = reader.GetOutput()

g = sv.parseGroupFileNew(args.group_file)

temp_data = []

for k,v in g.items():
    c = v['contour']

    n = c.shape[0]

    for i in range(n):
        p = c[i]
        d = {"group_id":k,
            "point_id":i,
            "model_id":args.model_id,
            "x":p[0],
            "y":p[1],
            "z":p[2]}

        temp_data.append(d)

for l in labels:
    data   = polyd.GetPointData().GetArray(l)

    tup_dim = len(data.GetTuple(0))

    for tn in range(tup_dim):
        new_label = l+'_'+str(tn)
        interp = sv.get_interp(polyd, l, tn)

        for i in range(len(temp_data)):
            d = temp_data[i]
            p = [d['x'], d['y'], d['z']]
            v = interp(p)[0]

            temp_data[i][new_label] = v

df = pd.DataFrame(temp_data)

path_name = args.group_file.split('/')[-1]

df.to_csv(args.output_dir+'/'+path_name+'_group.csv')
