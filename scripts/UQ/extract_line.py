import argparse
import os
import sys
import time
sys.path.append(os.path.abspath('../..'))
import pandas
import json
import modules.io as io
import modules.vascular_data as sv
parser = argparse.ArgumentParser()
parser.add_argument('-config')
parser.add_argument('-line_file')
parser.add_argument('-output_fn')

args = parser.parse_args()

cfg = io.load_json(args.config)

DIR          = cfg['dir']
MESH_LABELS  = cfg['mesh_labels']
GENS         = cfg['generations']
NUM_MODELS   = cfg['num_models']
SIM_NAME     = cfg['sim_name']
RESULTS_FILE = cfg['results_file']

LABS = cfg['labels']
LABELS = []
for i in range(cfg['start'], cfg['end'], cfg['incr']):
    for l in LABS:
        new_l = l+"_00"+str(i)
        LABELS.append(new_l)

LINE_FILE = io.load_json(args.line_file)
LINE       = LINE_FILE['line']
NUM_POINTS = len(LINE)

data = []

for gen in GENS:
    for mesh in MESH_LABELS:
        for n in range(NUM_MODELS):
            vtu_fn = DIR+'/'+str(gen)+'/'+mesh+'/'+str(n)+'/'+SIM_NAME+'/'+RESULTS_FILE
            print(vtu_fn)
            if not os.path.exists(vtu_fn): continue

            pd = sv.read_vtu(vtu_fn)
            v  = sv.probe_pd_line(pd,LINE)

            for j in range(NUM_POINTS):
                p = LINE[j]
                d = {"generation":gen,
                    "mesh":mesh,
                    "model":n,
                    "point":j,
                     "x":p[0],
                     "y":p[1],
                     "z":p[2]}

                for l in LABELS:
                    tup = v.GetPointData().GetArray(l).GetTuple(j)

                    for k in range(len(tup)):
                        d[l+'_'+str(k)] = tup[k]

                data.append(d)

df = pandas.DataFrame(data)
df.to_csv(args.output_fn)
