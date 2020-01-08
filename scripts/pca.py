import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

from modules import vascular_data as sv
from modules import io
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-config')

args = parser.parse_args()

config = io.load_json(args.config)

MODELS_DIR  = config['input_dir']
GROUP_FILES = config['group_files']
OUTPUT_DIR  = config['output_dir']
NUM_POINTS  = config['num_points']
MODES       = config['modes']
COEFFS      = config['coeffs']
GROUP_CONTENTS = config['group_contents_file']

case_dirs = os.listdir(MODELS_DIR)
case_dirs = [MODELS_DIR+'/'+f for f in case_dirs]

NUM_MODELS = len(case_dirs)

group_nums = {}
group_ids = {}
f = case_dirs[0]
tot_group_num = 0
for g_fn in GROUP_FILES:
    group = io.load_json(f+'/'+g_fn)
    group_ids[g_fn]  = group.keys()
    group_nums[g_fn] = len(group.keys())
    tot_group_num+=len(group.keys())

tot_point_num = NUM_POINTS*tot_group_num*3
MAT = np.zeros((NUM_MODELS,tot_point_num))

for i,g_dir in enumerate(case_dirs):
    pointer = 0
    for g_fn in GROUP_FILES:
        group = io.load_json(g_dir+'/'+g_fn)
        for j,k in enumerate(group.keys()):
            points = np.ravel(np.array(group[k]))
            pointer_end = pointer+NUM_POINTS*3
            MAT[i,pointer:pointer_end] = points
            pointer = pointer_end

mean_groups = np.mean(MAT,axis=0)
MAT_CENT    = MAT-mean_groups

U,S,V = np.linalg.svd(MAT_CENT,full_matrices=False)

np.save(OUTPUT_DIR+'/mean_groups.npy',mean_groups)
np.save(OUTPUT_DIR+'/U.npy',U)
np.save(OUTPUT_DIR+'/S.npy',S)
np.save(OUTPUT_DIR+'/V.npy',V)

mean_groups_dict = sv.vec_to_groups(mean_groups,GROUP_FILES,group_ids,group_nums,NUM_POINTS)
mean_dir = OUTPUT_DIR+'/mean_groups'
try:
    os.mkdir(mean_dir)
except:
    pass

for k,g in mean_groups_dict.items():
    io.write_json(g,mean_dir+'/'+k)
os.system("python segmentJsonToGroups.py -input_dir {}".format(mean_dir))
os.system("cp {} {}".format(GROUP_CONTENTS, mean_dir))

for i in range(MODES):
    for j,c in enumerate(COEFFS):
        print("vec {} coeff {} {}".format(i,j,c))
        vec = mean_groups + V[i]*c

        groups_dict = sv.vec_to_groups(vec,GROUP_FILES,group_ids,group_nums,NUM_POINTS)
        dir = OUTPUT_DIR+'/{}_{}'.format(i,j)
        try:
            os.mkdir(dir)
        except:
            pass

        for k,g in groups_dict.items():
            io.write_json(g,dir+'/'+k)
        os.system("python segmentJsonToGroups.py -input_dir {}".format(dir))
        os.system("cp {} {}".format(GROUP_CONTENTS, dir))
