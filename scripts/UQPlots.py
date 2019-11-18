import sys
import os
sys.path.append(os.path.abspath('..'))

import vtk
import modules.vascular_data as sv

fn    = "/home/marsdenlab/projects/SV/UQ/data/2_vessel/coarse/0/sim_steady/all_results.vtp"

label = "pressure_00700"

reader = vtk.vtkXMLPolyDataReader()
reader.SetFileName(fn)
reader.Update()

pd = reader.GetOutput()

########################
# Get Cell
########################
coord = [0.0]*3
tol   = 0.05

cell_id = sv.vtkPdFindCellId(pd, coord, tol)[0]
cell    = pd.GetCell(cell_id)
ids     = cell.GetPointIds()
print(ids.GetId(0), ids.GetId(1), ids.GetId(2))

data = pd.GetPointData().GetArray(label)
values = [data.GetValue(ids.GetId(i)) for i in range(3)]
