import argparse
import os
import sys
import time

sys.path.append(os.path.abspath('..'))

from modules import io_cluster as io

parser = argparse.ArgumentParser()
parser.add_argument('-config')

args = parser.parse_args()

cfg = io.load_json(args.config)

MAIN_DIR   = cfg['dir']
GENS       = cfg['generations']
MESHES     = cfg['mesh_labels']
NUM_MODELS = cfg['num_models']

CASE_DIR   = cfg['case_dir']
SIM_NAME   = cfg['sim_name']
VTU_FILE   = cfg['results_file']

for g in GENS:
    for i in range(NUM_MODELS):
        for me in MESHES:
            print(i,me)
            vtu_file = os.path.join(MAIN_DIR,str(g),me,str(i),SIM_NAME,VTU_FILE)

            out_vtu_file = "{}.vtu".format(i)

            out_file = os.path.join(CASE_DIR,str(g),me,"vtus",out_vtu_file)

            print(out_file)

            os.system("cp {} {}".format(vtu_file,out_file))
