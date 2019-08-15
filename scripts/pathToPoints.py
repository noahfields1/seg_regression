import argparse
import os
import sys

from modules import vascular_data as sv
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-path_file')
parser.add_argument('--interval', type=int, default=10)
parser.add_argument('-output_dir')

args = parser.parse_args()

path_dict = sv.parsePathFile(args.path_file)

for path_id in path_dict.keys():
    path_name = path_dict[path_id]['name']
    path_points = path_dict[path_id]['points']

    f = open(args.output_dir+'/'+path_name+'.txt','w')

    for i in range(0,len(path_points), args.interval):
        p = [str(x) for x in path_points[i]]
        s = str(i)+','+','.join(p)+'\n'
        f.write(s)
    f.close()
