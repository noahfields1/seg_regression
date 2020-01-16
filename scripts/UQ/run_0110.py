import os

CONFIGS = [
"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0110/extract/steady.json",
#"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0110/extract/steady_real.json",
"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0110/extract/steady_real_out.json"
#"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0110/extract/steady_real_in.json"
]

TUBES = [
'aorta',
'right_iliac'
]

print('tubes')

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube_new.py -config {} \
        -tube_file ../config_segment/0110/tubes/{}.json".format(cfg,p,name,p)
        )
