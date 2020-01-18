import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import sys
sys.path.append(os.path.abspath('..'))

from modules import vascular_data as sv
from modules import io
import numpy as np
from tqdm import tqdm

from sv_wrapper import SVWrapper

parser = argparse.ArgumentParser()
parser.add_argument('-image_file')
parser.add_argument('-points_file')
parser.add_argument('-output_dir')
parser.add_argument('-config')
parser.add_argument('-modes')

args = parser.parse_args()

if not os.path.isfile(args.config):
    raise RuntimeError("config file {} does not exist".format(
        args.config
    ))

if not os.path.isfile(args.image_file):
    raise RuntimeError("image file {} does not exist".format(
        args.image_file
    ))

net = SVWrapper(args.config)
net.set_image(args.image_file)

cfg = io.load_yaml(args.config)
if "DROPOUT_UQ" in cfg:
    print("sampling net with dropout p {}".format(cfg['DROPOUT_UQ']))
    net.model.sample(p=cfg['DROPOUT_UQ'])

p_file = open(args.points_file).readlines()
p_file = [p.replace('\n','') for p in p_file]

points = [np.array([float(x) for x in p.split(',')]) for p in p_file]

print( "saving groups file")
name = args.points_file.split('/')[-1].replace('.txt','')
print( name)
fn = args.output_dir+'/'+name+'.json'

data = {}
print("segmenting with {} fourier modes".format(args.modes))
for i,p in enumerate(points):
    print("{}, {}".format(name,i))
    c = net.segment_normal(p[1:4], p[7:], p[4:7], modes=args.modes)

    pos = int(p[0])

    data[pos] = c.tolist()

io.write_json(data,fn)
