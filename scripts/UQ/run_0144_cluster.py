import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-vtu')
args = parser.parse_args()


CONFIGS = [
"/home/gdmaher/seg_regression/scripts/config_segment/0144/extract/wom_rcr.json"
]

TUBES = [
'/home/gdmaher/seg_regression/scripts/config_segment/0144/tubes/aorta.json',
'/home/gdmaher/seg_regression/scripts/config_segment/0144/tubes/celiac_hepatic.json',
'/home/gdmaher/seg_regression/scripts/config_segment/0144/tubes/celiac_splenic.json',
'/home/gdmaher/seg_regression/scripts/config_segment/0144/tubes/ext_iliac_left.json',
'/home/gdmaher/seg_regression/scripts/config_segment/0144/tubes/renal_left.json',
'/home/gdmaher/seg_regression/scripts/config_segment/0144/tubes/renal_right.json',
'/home/gdmaher/seg_regression/scripts/config_segment/0144/tubes/SMA.json'
]


print('tubes')

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube_new_single.py -config {} \
        -tube_file {} \
        -vtu {}".format(cfg,p,args.vtu)
        )
