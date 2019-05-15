import argparse
import json
import os
import re
import sys
sys.path.append(os.path.abspath('..'))
from modules.vascular_data import normalizeContour

def get_regexp(line, field):
    path_name = re.search("{}=\"[A-Za-z_\-\+\.\:0-9 ]*\"".format(field), line)
    return path_name.group().split("\"")[1]

def parse_point(line):
    return [float(get_regexp(line,"x")),
        float(get_regexp(line,"y")), float(get_regexp(line,"z"))]


def parse_xml(input_file, output_dir, code):
    f = open(input_file,'r').readlines()
    f = [s.replace('\n','') for s in f]

    collecting = False

    #code = input_file.split('/')[-1].split('.')[0]

    for i,line in enumerate(f):
        if "<contourgroup" in line.lower():
            path_name = get_regexp(line, "path_name")
            path_id   = get_regexp(line, 'path_id')
        if "<contour " in line.lower():
            method = get_regexp(line, "method")
            c_type = get_regexp(line, "type")

        if "<path_point " in line:
            point_number = int(get_regexp(line, "id"))
            pos_line = f[i+1]
            tan_line = f[i+2]
            rot_line = f[i+3]
            i = i+4

            p = parse_point(pos_line)
            t = parse_point(tan_line)
            r = parse_point(rot_line)

        if "<contour_points>" in line:
            contour = []
            collecting = True
            continue
        if "</contour_points>" in line:
            collecting = False
            J = {}
            J['file']         = input_file
            J['path_name']    = path_name
            J['method']       = method
            J['point_number'] = point_number
            J['p']        = p
            J['tangent']  = t
            J['rotation'] = r
            J['contour3D']  = contour
            J['contour2D']  = normalizeContour(contour,p,t,r,as_list=True)
            J['code']     = code
            J['type']     = c_type
            output_filename = "{}.{}.{}.json".format(code,path_name,point_number)
            J['contour_name'] = output_filename

            output_filename = os.path.abspath(output_dir)+'/'+output_filename

            with open(output_filename,'w') as j:
                json.dump(J,j, sort_keys=True, indent=2)

        if collecting:
            point = parse_point(line)
            contour.append(point)
