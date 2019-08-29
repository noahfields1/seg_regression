import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

from modules import vascular_data as sv
from modules import io
import numpy as np
from tqdm import tqdm

from sv_wrapper import SVWrapper

parser = argparse.ArgumentParser()
parser.add_argument('-input_dir')
parser.add_argument('--deletions')
parser.add_argument('--corrections')

args = parser.parse_args()

groups_files = [f for f in os.listdir(args.input_dir) if '.json' in f\
    and not "_corrected" in f]
    
if len(groups_files)==0:
    raise RuntimeError("No json groups found")

names = [g.split('/')[-1].replace('.json','') for g in groups_files]
data  = [io.load_json(args.input_dir+'/'+g) for g in groups_files]

if args.deletions:
    if not os.path.isfile(args.deletions):
        raise RuntimeError("Could not find deletions file {}".format(args.deletions))

    deletions = io.load_json(args.deletions)

    for name,d in zip(names,data):
        if name in deletions:
            for k in deletions[name]:
                print("deleting {} {}".format(name,k))
                del d[k]

if args.corrections:
    if not os.path.isfile(args.corrections):
        raise RuntimeError("Could not find corrections file {}".format(args.corrections))

    corrections = io.load_json(args.corrections)

    for name,d in zip(names,data):
        if name in corrections:
            for k in corrections[name]:
                print("correcting {} {}".format(name,k))
                d[k] = corrections[name][k]

for name,d in zip(names,data):
    io.write_json(d,args.input_dir+'/'+name+'_corrected.json')
