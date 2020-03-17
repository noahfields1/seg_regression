import vtk
import sys
import os
import numpy as np
import subprocess

sys.path.append(os.path.abspath('..'))

from modules import vascular_data as sv

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-vtp')
parser.add_argument('-n',type=int)
parser.add_argument('-output_dir')

args = parser.parse_args()

for i in range(args.n):
    o = args.output_dir+'/{}.png'.format(i)
#    os.system("python screenvtk_array.py -vtp {} -label {} -output_fn {}".format(
#        args.vtp, i, o
#    ))

    p = subprocess.Popen(['python', 'screenvtk_array.py',
         '-vtp', args.vtp, '-label', str(i), '-output_fn', o])
    try:
        p.wait(3)
    except subprocess.TimeoutExpired:
        p.kill()
        print("moving on...")

    o = args.output_dir+'/{}_back.png'.format(i)

    p = subprocess.Popen(['python', 'screenvtk_array_back.py',
         '-vtp', args.vtp, '-label', str(i), '-output_fn', o])
    try:
        p.wait(3)
    except subprocess.TimeoutExpired:
        p.kill()
        print("moving on...")
