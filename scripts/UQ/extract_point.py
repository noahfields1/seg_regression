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
parser.add_argument('-point_file')
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
for i in range(cfg['start'], cfg['end']+cfg['incr'], cfg['incr']):
    for l in LABS:
        new_l = str(i)
        if len(new_l) == 1:
            new_l = "0000"+new_l
        if len(new_l) == 2:
            new_l = "000"+new_l
        if len(new_l) == 3:
            new_l = "00"+new_l
        if len(new_l) == 4:
            new_l = "0"+new_l
        LABELS.append(l+'_'+new_l)

POINTS_FILE = io.load_json(args.point_file)
p = POINTS_FILE['point']
data = []

for gen in GENS:
    for mesh in MESH_LABELS:
        for n in range(NUM_MODELS):
            vtu_fn = DIR+'/'+str(gen)+'/'+mesh+'/'+str(n)+'/'+SIM_NAME+'/'+RESULTS_FILE
            print(vtu_fn)
            if not os.path.exists(vtu_fn): continue
            d = {"mesh":mesh,
                "generation":gen,
                "model":n,
                 "x":p[0],
                 "y":p[1],
                 "z":p[2]}

            pd = sv.read_vtu(vtu_fn)
            v  = sv.probe_pd_point(pd,p)

            try:
                for l in LABELS:
                    tup = v.GetPointData().GetArray(l).GetTuple(0)

                    for j in range(len(tup)):
                        d[l+'_'+str(j)] = tup[j]

                data.append(d)
            except:
                pass
                
df = pandas.DataFrame(data)
df.to_csv(args.output_fn)
