import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-vtu')
args = parser.parse_args()

CONFIGS = [
"/home/gdmaher/seg_regression/scripts/config_segment/0110/extract/wom_rcr.json"
]

TUBES = [
'/home/gdmaher/seg_regression/scripts/config_segment/tubes/aorta.json',
'/home/gdmaher/seg_regression/scripts/config_segment/tubes/right_iliac.json',
]

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube_new_single.py -config {} \
        -tube_file {} \
        -vtu {}".format(cfg, p, args.vtu)
        )
