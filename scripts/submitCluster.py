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
    raise RuntimeError("Failed to load config {}".format(args.config))

RUNS_DIR    = cfg['runs_folder']

SIM_FOLDER = cfg['sim_folder']
SIM_NAME = SIM_DIR.split('/')[-1]

RUN_FILE   = cfg['run_file']
run_file_name = RUN_FILE.split('/')[-1]

run_folders = os.listdir(RUNS_DIR)
run_folders = [RUNS_DIR+'/'+f for f in run_folders]

for f in run_folders:
    print("running in {}".format(f))

    cur_sim_dir = f+'/'+SIM_NAME

    try:
        os.system("rm -r {}".format(cur_sim_dir))
    except:
        pass

    os.system("cp -r {} {}".format(SIM_DIR,f))

    os.system("cp {} {}".format(RUN_FILE, cur_sim_dir))

    os.chdir(cur_sim_dir)

    os.system("sbatch {}".format(run_file_name))

    time.sleep(0.3)
