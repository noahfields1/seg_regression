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

CASE_DIR     = cfg['case_dir']
SIM_DIR      = cfg['sim_dir']
RESULTS_FILE = cfg['results_file']
OUTPUT_DIR   = cfg['output_dir']
LABELS_FILE  = cfg['labels_file']
COORD_FILES  = cfg['coord_files']

dirs     = os.listdir(CASE_DIR)

models = [CASE_DIR+'/'+d+'/'+SIM_DIR+'/'+RESULTS_FILE for d in dirs]

for o,m in zip(dirs,models):
    for c in COORD_FILES:
        print(m,c)
        os.system('python UQextract_coord.py -vtu {} -coord_file {} -labels_file {} -output_dir {} -model_id {}'.format(
            m, c, LABELS_FILE, OUTPUT_DIR+'/'+o, o
        ))
