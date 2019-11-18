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
