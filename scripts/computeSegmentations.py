import argparse
import os
import sys


from modules import vascular_data as sv
import numpy as np
from tqdm import tqdm

from sv_wrapper import SVWrapper

parser = argparse.ArgumentParser()
parser.add_argument('-image_file')
parser.add_argument('-points_file')
parser.add_argument('-output_dir')
parser.add_argument('-config')

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

p_file = open(args.points_file).readlines()
p_file = [p.replace('\n','') for p in p_file]

points = [np.array([float(x) for x in p.split(',')]) for p in p_file]

print( "saving groups file")
name = args.points_file.split('/')[-1].replace('.txt','')
print( name)
f = open(args.output_dir+'/'+name,'w')

for i,p in enumerate(points):

    c = net.segment_normal(p[:3], p[3:6], p[6:])

    pos = int(p[0])
    f.write('/group/{}/{}\n'.format(name,pos))
    f.write(str(pos) +'\n')
    f.write('posId {}\n'.format(pos))
    for j in range(c.shape[0]):
        f.write('{} {} {}\n'.format(c[j][0],c[j][1],c[j][2]))
    f.write('\n')
f.close()
