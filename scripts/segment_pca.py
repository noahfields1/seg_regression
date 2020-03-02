import argparse
import os
import sys
import subprocess
sys.path.append(os.path.abspath('..'))

import modules.io as io

parser = argparse.ArgumentParser()
parser.add_argument('-config')

args = parser.parse_args()

cfg = io.load_json(args.config)

dirs = cfg['PCA_DIRS']
edge_size = 0.1

for d in dirs:
    os.system('python segmentMesh.py -config {}\
     -input_dir {} -output_dir {} -edge_size {}'.format(args.config,d,d,edge_size))

    os.system('python segmentMeshMerge.py -config {}\
     -input_dir {} -output_dir {} -edge_size {}'.format(args.config,d,d,edge_size))

    os.system('python segmentMesh_smooth.py -config {}\
     -input_dir {} -output_dir {} -edge_size {}'.format(args.config,d,d,edge_size))

    # os.system('python screenvtk.py -vtp {}/exterior.vtp\
    #  -output_fn {}/exterior.png'.format(d,d))

    p = subprocess.Popen(['python', 'screenvtk.py',
         '-vtp', d+'/exterior.vtp', '-output_fn', d+'/exterior.png'])
    try:
        p.wait(3)
    except subprocess.TimeoutExpired:
        p.kill()
        print("moving on...")
