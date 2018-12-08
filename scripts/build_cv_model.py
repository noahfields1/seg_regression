import os
import sys
sys.path.append(os.path.abspath('..'))

import argparse

import modules.sv_image as sv_image
import modules.vascular_data as sv
import modules.io as io
import factories.model_factory as model_factory

parser = argparse.ArgumentParser()

parser.add_argument('-i', description='medical image file')
parser.add_argument('-p', description='.paths file')
parser.add_argument('-c', description='model config file')
parser.add_argument('-o', description='output directory')

args = parser.parse_args()

image     = sv_image.Image(args.i)
path_dict = sv.parsePathFile(args.p)
config    = io.load_yaml(args.c)

output_dir = os.path.abspath(args.o)
name_dir   = os.path.join(output_dir, config['NAME'])
group_dir  = os.path.join(name_dir, 'groups')

model = model_factory.get_model(config)

for p in [output_dir, name_dir, group_dir]:
    if not os.path.isdir(p): os.mkdir(p)
