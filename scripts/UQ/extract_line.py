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
parser.add_argument('-vtu_list')
parser.add_argument('-line_file')
parser.add_argument('-label_file')
parser.add_argument('-output_fn')

args = parser.parse_args()

VTUS = open(args.vtu_list,'r').readlines()
VTUS = [v.replace('\n','') for v in VTUS]

LABELS = open(args.label_file,'r').readlines()
LABELS = [v.replace('\n','') for v in LABELS]

LINE_FILE = io.load_json(args.line_file)
LINE = LINE_FILE['line']
NUM_POINTS = len(LINE)
data = []

for i,vtu_fn in enumerate(VTUS):
    if not os.path.exists(vtu_fn): continue

    pd = sv.read_vtu(vtu_fn)
    v  = sv.probe_pd_line(pd,LINE)

    for j in range(NUM_POINTS):
        p = LINE[j]
        d = {"model":i,
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
