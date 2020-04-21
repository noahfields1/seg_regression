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

RUNS_DIRS    = cfg['runs_folders']

SIM_NAME = cfg['sim_name']

RUN_FILE   = cfg['run_file']

run_file_name = RUN_FILE.split('/')[-1]

run_folders = []
for d in RUNS_DIRS:
    rf = os.listdir(d)
    rf = [str(d+'/'+f) for f in rf]
    run_folders+=rf

if "num_models" in cfg:
    n = cfg['num_models']
else:
    n = len(run_folders)

for i,f in enumerate(run_folders[:n]):
    print(f)

    try:
    	cur_sim_dir = f+'/'+SIM_NAME

    	os.system("cp {0} {1}".format(RUN_FILE, cur_sim_dir))

    	os.chdir(cur_sim_dir)

    	os.system("sbatch {0}".format(run_file_name))

    	time.sleep(0.3)
    except:
	print("failed")
