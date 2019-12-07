import os

MESHES = ['coarse','medium','fine']
POINTS = ['inlet',
'outlet_aorta',
'outlet_right_iliac',
'bifurcation',
'bifurcation_aorta',
'bifurcation_right_iliac']

LINES = [
'aorta_center',
'right_iliac_center',
]

print('points')
for m in MESHES:
    for p in POINTS:
        print(m,p)
        os.system(
            "python extract_point.py -vtu_list ../config_segment/0110/vtu_{}.txt \
             -label_file ../config_segment/0110/labels.txt \
             -point_file ../config_segment/0110/points/{}.json \
             -output_fn /home/marsdenlab/projects/SV/UQ/data/2_vessel_data/data/{}/{}.csv".format(m,p,m,p)
             )

print('lines')
for m in MESHES:
    for p in LINES:
        print(m,p)
        os.system("python extract_line.py -vtu_list ../config_segment/0110/vtu_{}.txt \
        -label_file ../config_segment/0110/labels.txt \
        -line_file ../config_segment/0110/lines/{}.json \
        -output_fn /home/marsdenlab/projects/SV/UQ/data/2_vessel_data/data/{}/{}.csv".format(m,p,m,p)
        )
