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

SIM_DIR = cfg['sim_folder']
SIM_NAME = SIM_DIR.split('/')[-1]

RUN_FILE   = cfg['run_file']
NBATCH     = cfg['num_batch']
SLEEP_BATCH = cfg['sleep_batch']

run_file_name = RUN_FILE.split('/')[-1]

run_folders = []
for d in RUNS_DIRS:
    rf = os.listdir(d)
    rf = [str(d+'/'+f) for f in rf]
    run_folders+=rf

for i,f in enumerate(run_folders):
    print(f)
#    print("running in {}".format(f))

    cur_sim_dir = f+'/'+SIM_NAME

    try:
        os.system("rm -r {0}".format(cur_sim_dir))
    except:
        pass

    os.system("cp -r {0} {1}".format(SIM_DIR,f))

    os.system("cp {0} {1}".format(RUN_FILE, cur_sim_dir))

    os.chdir(cur_sim_dir)

    os.system("sbatch {0}".format(run_file_name))

    time.sleep(0.3)

    if i%NBATCH == 0:
        time.sleep(SLEEP_BATCH)
    #import pdb; pdb.set_trace()
