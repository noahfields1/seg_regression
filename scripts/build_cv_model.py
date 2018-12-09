import os
import sys
sys.path.append(os.path.abspath('..'))

import argparse

import modules.sv_image as sv_image
import modules.vascular_data as sv
import modules.io as io
import factories.model_factory as model_factory

parser = argparse.ArgumentParser()

parser.add_argument('-i', help='medical image file')
parser.add_argument('-p', help='.paths file')
parser.add_argument('-c', help='model config file')
parser.add_argument('-o', help='output directory')

args = parser.parse_args()

image     = sv_image.Image(args.i)
path_dict = sv.parsePathFile(args.p)
config    = io.load_yaml(args.c)

image.set_spacing(config['SPACING'])
image.set_reslice_ext(config['CROP_DIMS'])

output_dir = os.path.abspath(args.o)
name_dir   = os.path.join(output_dir, config['NAME'])
group_dir  = os.path.join(name_dir, 'groups')

model = model_factory.get_model(config)
model.load()
for p in [output_dir, name_dir, group_dir]:
    if not os.path.isdir(p): os.mkdir(p)

for path_id, path in path_dict.items():
        path_name = path['name']
        path_points = path['points']

        p = [x[:3] for x in path_points]
        n = [x[3:6] for x in path_points]
        v = [x[6:] for x in path_points]

        X = image.get_reslices(p,n,v)

        C = model.predict(X)*config['CROP_DIMS']/2

        path_dict[path_id]['contours_2d'] = C

        C3 = [sv.denormalizeContour(c_,p_,n_,v_) for c_,p_,n_,v_ in zip(C,p,n,v)]

        path_dict[path_id]['contours_3d'] = C3

for path_id, path in path_dict.items():
    #TODO write to files
    #TODO write groups toc file
