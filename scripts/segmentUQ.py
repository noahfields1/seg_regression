import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

parser = argparse.ArgumentParser()
parser.add_argument('-input_dir')
parser.add_argument('-output_dir')
parser.add_argument('-config')
parser.add_argument('--n', type=int, default=10)

args = parser.parse_args()
DIR = os.path.abspath(args.input_dir)

if not os.path.isdir(args.input_dir):
    raise RuntimeError("input dir doesnt exist")

if not os.path.isdir(args.output_dir):
    raise RuntimeError("input dir doesnt exist")

if not os.path.isfile(args.config):
    raise RuntimeError("config doesnt exist")


for i in range(args.n):
    print("\nSEGMENTING {}\n".format(i))

    odir = os.path.abspath(args.output_dir)+'/'+str(i)
    try:
        os.mkdir(odir)
    except:
        pass

    os.system('python segmentAll.py\
         -input_dir {} -output_dir {} -config {}'.format(
            args.input_dir, odir, args.config))
