import os

CONFIGS = [
#"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0110/extract/steady.json",
#"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0110/extract/steady_real.json",
"/home/marsdenlab/projects/seg_regression/scripts/config_segment/0110/extract/wom_rcr.json"
]

# POINTS = ['inlet',
# 'aorta',
# 'outlet_aorta',
# 'outlet_right_iliac',
# 'bifurcation',
# 'bifurcation_aorta',
# 'bifurcation_right_iliac']
#
# LINES = [
# 'aorta_center',
# 'right_iliac_center',
# ]

TUBES = [
'aorta_single',
'aorta_single_2',
'right_iliac_single'
]

print('points')

# for cfg in CONFIGS:
#     for p in POINTS:
#         print(cfg,p)
#         name = cfg.split('/')[-1].replace('.json','')
#         os.system(
#             "python extract_point.py -config {} \
#              -point_file ../config_segment/0110/points/{}.json \
#              -output_fn /media/marsdenlab/Data1/UQ/0110/csv/{}/{}.csv".format(cfg,p,name,p)
#              )

# print('lines')
#
# for cfg in CONFIGS:
#     for p in LINES:
#         print(cfg,p)
#         name = cfg.split('/')[-1].replace('.json','')
#         os.system("python extract_line.py -config {} \
#         -line_file ../config_segment/0110/lines/{}.json \
#         -output_fn /media/marsdenlab/Data1/UQ/0110/csv/{}/{}.csv".format(cfg,p,name,p)
#         )

print('tubes')

for cfg in CONFIGS:
    for p in TUBES:
        print(cfg, p)
        name = cfg.split('/')[-1].replace('.json','')

        os.system("python extract_tube.py -config {} \
        -tube_file ../config_segment/0110/tubes/{}.json \
        -output_fn /media/marsdenlab/Data1/UQ/0110/csv/{}/{}_tube.csv".format(cfg,p,name,p)
        )
