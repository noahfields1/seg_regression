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

OUTPUT_DIR = cfg["OUTPUT_DIR"]
IMAGE      = cfg["IMAGE"]
PATHS      = cfg["PATH_FILES"]
NN_CONFIG  = cfg["NN_CONFIG"]
NAMES      = cfg["NAMES"]

DEL_FILE   = cfg["DELETE_FILE"]
COR_FILE   = cfg["CORRECT_FILE"]

INTERVAL   = cfg["INTERVAL"]
NUM_RUNS   = cfg["NUM_RUNS"]

if not os.path.isdir(OUTPUT_DIR):
    raise RuntimeError("output dir doesnt exist")

if not os.path.isfile(NN_CONFIG):
    raise RuntimeError("config doesnt exist")

for i in range(NUM_RUNS):
    print("\nSEGMENTING {}\n".format(i))

    odir = OUTPUT_DIR+'/'+str(i)
    try:
        os.mkdir(odir)
    except:
        pass

    os.system('python segmentAll.py\
         -config {} -output_dir {}'.format(
            args.config, odir)
            )

    os.system('python segmentCorrect.py\
         -input_dir {} --deletions {} --corrections {}'.format(
            odir, DEL_FILE, COR_FILE)
            )

    os.system('python segmentJsonToGroups.py\
         -input_dir {}'.format(odir)
         )

    os.system('python segmentMesh.py\
         -config {} -output_dir {}'.format(args.config, odir)
         )

    os.system('python segmentMeshComplete.py\
         -config {} -output_dir {}'.format(args.config, odir)
         )
