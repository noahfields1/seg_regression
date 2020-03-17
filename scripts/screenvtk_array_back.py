import vtk
import sys
import os
import numpy as np

sys.path.append(os.path.abspath('..'))

from modules import vascular_data as sv

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-vtp')
parser.add_argument('-label')
parser.add_argument('-output_fn')

args = parser.parse_args()

# create source
pd = sv.vtk_read_native_polydata(args.vtp)
data = pd.GetPointData().GetArray(args.label)

N = data.GetNumberOfTuples()
v = []
for i in range(N):
    x = data.GetTuple(i)[0]
    v.append(x)

ma = np.amax(v)
mi = np.amin(v)

colors = vtk.vtkUnsignedCharArray()
colors.SetNumberOfComponents(3)
colors.SetName("Colors")

for i in range(N):
    x = (v[i]-mi)*1.0/(ma-mi)
    xa = (255*(x),255*(1.0-x),0)
    colors.InsertNextTuple(xa)

pd.GetPointData().SetScalars(colors)
# mapper
mapper = vtk.vtkPolyDataMapper()
mapper.SetInputDataObject(pd)

# actor
actor = vtk.vtkActor()
actor.SetMapper(mapper)

# color the actor
#actor.GetProperty().SetColor(1,0,0) # (R,G,B)

# create a rendering window and renderer
ren = vtk.vtkRenderer()
ren.SetBackground(1,1,1)

camera = vtk.vtkCamera();

bounds = pd.GetBounds()
x = (bounds[0]+bounds[1])*1.0/2
y = (bounds[2]+bounds[3])*1.0/2
z = (bounds[4]+bounds[5])*1.0/2

h = (bounds[5]-bounds[4])

offset = 1.5*h
print(x,y,z,h)

camera.SetPosition(x, -int(y-2*offset), int(z))
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
