import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-vtu')
args = parser.parse_args()

CONFIGS = [
"/home/gdmaher/seg_regression/scripts/config_segment/coronary/extract/wom_rcr.json"
]

TUBES = [
'/home/gdmaher/seg_regression/scripts/config_segment/coronary/tubes/lc1.json',
'/home/gdmaher/seg_regression/scripts/config_segment/coronary/tubes/lc1_sub1.json',
'/home/gdmaher/seg_regression/scripts/config_segment/coronary/tubes/lc1_sub2.json',
'/home/gdmaher/seg_regression/scripts/config_segment/coronary/tubes/lc1_sub3.json',
'/home/gdmaher/seg_regression/scripts/config_segment/coronary/tubes/lc2.json',
'/home/gdmaher/seg_regression/scripts/config_segment/coronary/tubes/lc2_sub1.json'
]


print('tubes')

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube_new.py -config {} \
        -tube_file {} \
        -vtu {}".format(cfg,p,args.vtu)
        )
