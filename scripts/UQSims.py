import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

import modules.io as io

parser = argparse.ArgumentParser()
parser.add_argument('-config')

args = parser.parse_args()

try:
    cfg = io.load_json(args.config)
except:
    raise RuntimeError("Failed to load config {}".format(args.config))

RUNS_DIR = cfg['runs_folder']
SIM_DIR = cfg['sim_folder']
SIM_NAME = SIM_DIR.split('/')[-1]

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

    os.chdir(cur_sim_dir)

    os.system("bash run.sh")
