import vtk
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('..'))

from modules import vascular_data as sv

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-vtp')
parser.add_argument('-vtp2')
parser.add_argument('-output_fn')

args = parser.parse_args()

# create source
pd = sv.vtk_read_native_polydata(args.vtp)

# mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputDataObject(pd)

# actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

#VTP 2
# create source
pd2 = sv.vtk_read_native_polydata(args.vtp2)

# mapper
mapper2 = vtk.vtkPolyDataMapper()
mapper2.SetInputDataObject(pd2)

# actor
actor2 = vtk.vtkActor()
actor2.SetMapper(mapper2)

# create a rendering window and renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1,1,1)

camera = vtk.vtkCamera();

bounds = pd.GetBounds()
x = (bounds[0]+bounds[1])*1.0/2
y = (bounds[2]+bounds[3])*1.0/2
z = (bounds[4]+bounds[5])*1.0/2

h = (bounds[5]-bounds[4])

offset = 1.75*h
print(x,y,z,h)

camera.SetPosition(x, -int(y+offset), int(z))
camera.SetFocalPoint(x, y, z);

ren.SetActiveCamera(camera)

renWin = vtk.vtkRenderWindow()
renWin.SetSize(800,800)
renWin.AddRenderer(ren)

# create a renderwindowinteractor
iren = vtk.vtkRenderWindowInteractor()
iren.SetRenderWindow(renWin)

# assign actor to the renderer
ren.AddActor(actor)
ren.AddActor(actor2)

renWin.Render()

# screenshot code:
w2if = vtk.vtkWindowToImageFilter()
w2if.SetInput(renWin)
w2if.Update()

writer = vtk.vtkPNGWriter()
writer.SetFileName(args.output_fn)
writer.SetInputData(w2if.GetOutput())
writer.Write()

# enable user interface interactor
iren.Initialize()
iren.Start()
