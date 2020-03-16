import os

CONFIGS = [
#"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0144/extract/wom_rcr.json",
"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0144/extract/wom_rcr_fixed.json"
]

TUBES = [
'aorta',
'celiac_hepatic',
'celiac_splenic',
'ext_iliac_left',
'renal_left',
'renal_right',
'SMA'
]


print('tubes')

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube_new.py -config {} \
        -tube_file ../config_segment/0144/tubes/{}.json".format(cfg,p)
        )
