import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-path')
parser.add_argument('-output_fn')
args = parser.parse_args()

path = open(args.path,'r').readlines()

path = [p.replace('\n','') for p in path]
path = [p.split(',') for p in path]

d = {}
d['points']   = [ [float(p[1]),float(p[2]),float(p[3])] for p in path]
d['normals']  = [ [float(p[4]),float(p[5]),float(p[6])] for p in path]
d['radiuses'] = [1.0 for i in range(len(path))]


with open(args.output_fn,'w') as f:
    json.dump(d,f, indent=2)
