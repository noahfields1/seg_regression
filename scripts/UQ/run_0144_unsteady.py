import os

CONFIGS = [
"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0144/extract/wom_rcr.json"
]

TUBES = [
'aorta_outlet'
#'aorta_single_1',
# 'aorta_single_2',
# 'aorta_single_3',
# 'aorta_single_4',
# 'aorta_single_5',
# "celiac_hepatic_1",
# "celiac_hepatic_2",
# "celiac_splenic_1",
# "celiac_splenic_2",
# "ext_iliac_left_single_1",
# "ext_iliac_left_single_2",
# "renal_left_1",
# "renal_left_2",
# "renal_right_1",
# "renal_right_2",
# "SMA_1",
# "SMA_2"
]


print('tubes')

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube_new.py -config {} \
        -tube_file ../config_segment/0144/tubes/{}.json".format(cfg,p)
        )
