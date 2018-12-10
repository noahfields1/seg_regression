import os
import sys
sys.path.append(os.path.abspath('..'))

import argparse

import modules.sv_image as sv_image
import modules.vascular_data as sv
import modules.io as io
import factories.model_factory as model_factory
import modules.vessel_regression as vessel_regression

import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('-i', help='medical image file')
parser.add_argument('-p', help='.paths file')
parser.add_argument('-c', help='model config file')
parser.add_argument('-o', help='output directory')
parser.add_argument('-n', help='identifier')
args = parser.parse_args()

image     = sv_image.Image(args.i)
path_dict = sv.parsePathFile(args.p)
config    = io.load_yaml(args.c)

image.set_spacing(config['SPACING'])
image.set_reslice_ext(config['CROP_DIMS'])

output_dir = os.path.abspath(args.o)
name_dir   = os.path.join(output_dir, config['NAME'])
group_dir  = os.path.join(name_dir, 'groups')
model_dir  = os.path.join(group_dir, args.n)

model = model_factory.get_model(config)
model.load()
for p in [output_dir, name_dir, group_dir, model_dir]:
    if not os.path.isdir(p): os.mkdir(p)

for path_id, path in path_dict.items():
    path_name = path['name']
    path_points = path['points']

    p = [x[:3] for x in path_points]
    n = [x[3:6] for x in path_points]
    v = [x[6:] for x in path_points]

    X = image.get_reslices(p,n,v)
    mu = np.mean(X,axis=(1,2))[:,np.newaxis,np.newaxis]
    std = np.std(X,axis=(1,2))[:,np.newaxis,np.newaxis]
    X = (1.0*X-mu)/(std+1e-3)

    C = model.predict(X)
    C = np.array([vessel_regression.pred_to_contour(c) for c in C])
    C = C*config['CROP_DIMS']*config['SPACING']/2

    path_dict[path_id]['contours_2d'] = C

    C3 = np.array([sv.denormalizeContour(c_,p_,n_,v_) for c_,p_,n_,v_ in zip(C,p,n,v)])

    path_dict[path_id]['contours_3d'] = C3

for path_id, path in path_dict.items():

    print( "saving groups file")
    name   = path['name']
    points = path['points']
    print( name)

    f = open(os.path.join(model_dir, name),'w')

    for i,p in enumerate(points):
        c = path['contours_3d'][i]
        pos = i
        f.write('/group/{}/{}\n'.format(name,pos))
        f.write(str(pos) +'\n')
        f.write('posId {}\n'.format(pos))
        for j in range(c.shape[0]):
            f.write('{} {} {}\n'.format(c[j][0],c[j][1],c[j][2]))
        f.write('\n')

    f.close()
