'''
This script generates a mesh using radius based meshing, a method based on the
distance to the solid model's centrline.
'''
import sv
#import sv_vis
import os
import sys
import math

def calculate_centerlines(model_name, model_polydata_name, source_ids, target_ids):
    '''
    Calculate centerlines and the distance to centerlines array.
    The distance to centerlines is stored in polydata referenced by 'dist_name'.
    '''
    lines_name = model_name + "_lines"
    sep_lines_name = model_name + "_sep_lines"
    voronoi_name = model_name + "_voronoi"
    dist_name = model_name + "_distance"
    sv.VMTKUtils.Centerlines(model_polydata_name, source_ids, target_ids, lines_name, voronoi_name)
    sv.VMTKUtils.Separatecenterlines(lines_name, sep_lines_name)
    sv.VMTKUtils.Distancetocenterlines(model_polydata_name, sep_lines_name, dist_name)
    # Display the centerlines.
    # lines_actor = sv_vis.pRepos(renderer, sep_lines_name)[1]
    # lines_actor.GetProperty().SetColor(0,1,0)
    # lines_file_name = lines_name + '.vtp'
    # sv.Repository.WriteVtkPolyData(sep_lines_name, "ascii", lines_file_name)

    dist_pd = sv.Repository.ExportToVtk(dist_name)
    import pdb; pdb.set_trace()
    dist_array = dist_pd.GetPointData().GetArray('DistanceToCenterlines')
    dist_range = 2*[0.0]
    dist_array.GetRange(dist_range, 0)
    print("Minumum distance: {0:f}".format(dist_range[0]))

    return dist_name

def read_solid_model(model_name,solid_file_name):
    '''
    Read in a solid model.
    '''
    sv.Solid.SetKernel('PolyData')
    solid = sv.Solid.pySolidModel()
    solid.ReadNative(model_name, solid_file_name)
    solid.GetBoundaryFaces(60)
    print ("Model face IDs: " + str(solid.GetFaceIds()))
    model_polydata_name = model_name + "_pd"
    solid.GetPolyData(model_polydata_name)
    # model_actor = sv_vis.pRepos(renderer, model_polydata_name)[1]
    # model_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    # #sv_vis.polyDisplayWireframe(renderer, model_polydata)
    # sv_vis.polyDisplayPoints(renderer, model_polydata_name)

    return solid, model_polydata_name, solid_file_name
