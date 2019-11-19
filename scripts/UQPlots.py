import sys
import os
sys.path.append(os.path.abspath('..'))

import vtk
import modules.vascular_data as sv
import modules.io as io

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

fn    = "/home/marsdenlab/projects/SV/UQ/data/2_vessel/coarse/0/sim_steady/all_results.vtu"

g_fn  = "/home/marsdenlab/projects/SV/UQ/data/2_vessel/coarse/0/aorta_corrected.json"
group = io.load_json(g_fn)
centerline = []
for k in group:
    c = np.array(group[k])
    p = np.mean(c,axis=0)
    centerline.append(p)

path_fn = "/home/marsdenlab/projects/SV/UQ/data/paths/aorta.txt"
path    = sv.parsePathPointsFile(path_fn)

label = "velocity_00700"

reader = vtk.vtkXMLUnstructuredGridReader()
reader.SetFileName(fn)
reader.Update()

pd = reader.GetOutput()


##########################################
#interpolate
##########################################
N      = pd.GetNumberOfPoints()
points = pd.GetPoints()
data   = pd.GetPointData().GetArray(label)

points_arr = []
data_arr = []
for i in range(N):
    p = points.GetPoint(i)
    d = data.GetTuple(i)[0]

    points_arr.append(p)
    data_arr.append(d)

points_arr = np.array(points_arr)
data_arr   = np.array(data_arr)

interp = LinearNDInterpolator(points_arr, data_arr)

vals = []

for p in path:
    coord = p[1:4]
    v = interp(coord)
    print(coord,v)
    vals.append(v)

vals_center = []
for p in centerline:
    v = interp(p)
    print(p,v)
    vals_center.append(v)

plt.figure()
plt.plot(vals, color='b', label='path')
plt.plot(vals_center, color='r', label='center')
plt.show()
