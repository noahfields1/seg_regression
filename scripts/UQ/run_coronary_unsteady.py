import os

CONFIGS = [
"/home/marsdenlab/projects/seg_regression/scripts/config_segment/coronary/extract/wom_rcr_converge.json"
]

TUBES = [
'lc1'
#'lc1_sub1',
#'lc1_sub2',
#'lc1_sub3',
#'lc2',
#'lc2_sub1'
]


print('tubes')

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube_new.py -config {} \
        -tube_file ../config_segment/coronary/tubes/{}.json".format(cfg,p)
        )
