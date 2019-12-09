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

PD = "/home/marsdenlab/projects/SV/UQ/data/2_vessel_consistent/coarse/8/sim_steady_highr/all_results.vtu"

pd = sv.read_vtu(PD)

p = [-0.088387, 0.967268, 0.923072]
n = [-0.072288, 0.227461, -0.971101]
r = 2.0

surf = sv.clip_plane_rad(pd, p, n, r)

featureEdges = vtk.vtkFeatureEdges()
featureEdges.SetInputData(surf)
featureEdges.BoundaryEdgesOn()
featureEdges.ManifoldEdgesOff()
featureEdges.NonManifoldEdgesOff()
featureEdges.Update()

edge_pd = featureEdges.GetOutput()

sv.vtk_write_polydata(surf,'surf.vtp')
sv.vtk_write_polydata(edge_pd,'edge.vtp')

nlines = edge_pd.GetNumberOfLines()

for i in range(nlines):
    line = edge_pd.GetCell(i)
    pt_ids = line.GetPointIds()

    for j in range(2):
        id = pt_ids.GetId(j)
        print(i,j,id)
