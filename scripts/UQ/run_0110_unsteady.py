import os

CONFIGS = [
"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0110/extract/wom_rcr.json"
]

TUBES = [
'aorta_outlet',
# 'aorta_single_1',
# 'aorta_single_2',
# 'aorta_single_3',
# 'right_iliac_single',
# 'right_iliac_single_1',
# 'right_iliac_single_2',
]


print('tubes')

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube_new.py -config {} \
        -tube_file ../config_segment/0110/tubes/{}.json".format(cfg,p)
        )
