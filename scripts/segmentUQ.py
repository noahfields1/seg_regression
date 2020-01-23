import argparse
import os
import sys
sys.path.append(os.path.abspath('..'))

import modules.io as io
import subprocess

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
EDGE_SIZES = cfg["EDGE_SIZES"]
TIMEOUT    = 15*60

if not os.path.isdir(OUTPUT_DIR):
    raise RuntimeError("output dir doesnt exist")

if not os.path.isfile(NN_CONFIG):
    raise RuntimeError("config doesnt exist")

for i in range(NUM_RUNS):
    print("\nSEGMENTING {}\n".format(i))

    od   = OUTPUT_DIR+'/models'
    odir = od+'/'+str(i)
    try:
        os.mkdir(od)
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

    for k,e in EDGE_SIZES.items():

        ed   = OUTPUT_DIR + '/' + k
        edir = ed + '/' + str(i)
        print(edir)
        try:
            os.mkdir(ed)
        except:
            pass

        try:
            os.mkdir(edir)
        except:
            pass

        os.system('python segmentMesh.py\
             -config {} -input_dir {} -output_dir {} -edge_size {}'.format(
                args.config, odir,edir,e)
             )

        # p = subprocess.Popen(['python', 'segmentMesh.py',
        #      '-config', args.config, '-input_dir', odir, '-output_dir',
        #      edir, '-edge_size', str(e)])
        # try:
        #     p.wait(TIMEOUT)
        # except subprocess.TimeoutExpired:
        #     p.kill()
        #     print(k,e,"FAILED")
        #     break

        # os.system('python segmentMeshMerge.py\
        #      -config {} -input_dir {} -output_dir {} -edge_size {}'.format(
        #         args.config, odir,edir,e)
        #      )

        p = subprocess.Popen(['python', 'segmentMeshMerge.py',
             '-config', args.config, '-input_dir', odir, '-output_dir',
             edir, '-edge_size', str(e)])
        try:
            p.wait(TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill()
            print(k,e,"FAILED")
            break

        # os.system('python segmentMesh2.py\
        #      -config {} -input_dir {} -output_dir {} -edge_size {}'.format(
        #         args.config, odir,edir,e)
        #      )

        p = subprocess.Popen(['python', 'segmentMesh2.py',
             '-config', args.config, '-input_dir', odir, '-output_dir',
             edir, '-edge_size', str(e)])
        try:
            p.wait(TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill()
            print(k,e,"FAILED")
            break

        os.system('python segmentCapIds.py\
             -config {} -output_dir {}'.format(
                args.config, edir)
             )

        p = subprocess.Popen(['python', 'segmentMesh3.py',
             '-config', args.config, '-input_dir', odir, '-output_dir',
             edir, '-edge_size', str(e)])
        try:
            p.wait(TIMEOUT)
        except subprocess.TimeoutExpired:
            p.kill()
            print(k,e,"FAILED")
            break

        os.system('python segmentMeshComplete.py\
             -config {} -output_dir {}'.format(args.config, edir)
             )
