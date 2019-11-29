import argparse
import os
import sys
import time
sys.path.append(os.path.abspath('..'))

import json

parser = argparse.ArgumentParser()
parser.add_argument('-config')

args = parser.parse_args()

try:
    with open(args.config) as f:
        cfg = json.load(f)
except:
    raise RuntimeError("Failed to load config {0}".format(args.config))

RUNS_DIR    = cfg['runs_folder']

SIM_DIR = cfg['sim_folder']
SIM_NAME = SIM_DIR.split('/')[-1]

RUN_FILE   = cfg['run_file']

run_folders = os.listdir(RUNS_DIR)
run_folders = [str(RUNS_DIR+'/'+f) for f in run_folders]


for f in run_folders:
    print(f)
#    print("running in {}".format(f))

    cur_sim_dir = f+'/'+SIM_NAME

    os.chdir(cur_sim_dir)

    os.system("bash {0}".format(RUN_FILE))

    time.sleep(0.3)

    #import pdb; pdb.set_trace()
