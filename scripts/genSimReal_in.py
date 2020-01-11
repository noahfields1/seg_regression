import argparse
import os
import sys
import time
import numpy as np

sys.path.append(os.path.abspath('..'))

from modules import io
from modules import vascular_data as sv

parser = argparse.ArgumentParser()
parser.add_argument('-config')

args = parser.parse_args()

cfg = io.load_json(args.config)

GEN_DIRS   = cfg['gen_dirs']
MESH_DIRS  = cfg['mesh_dirs']
MODEL_DIRS = [f+'/models' for f in GEN_DIRS]
NUM_MODELS = cfg['num_models']

SIM_DIR    = cfg['sim_dir']
SIM_NAME   = SIM_DIR.split('/')[-1]

GROUPS       = cfg['groups']
GROUPS_FILES = cfg['groups_files']
TOTAL_R      = cfg['total_r']

SOLVER_FILE  = cfg['solver_file']
SOLVER_LINES = open(SIM_DIR+'/'+SOLVER_FILE,'r').readlines()

INLET = cfg['inlet']

for g,mo in zip(GEN_DIRS,MODEL_DIRS):
    for i in range(NUM_MODELS):

        g_inlet = io.load_json(mo+'/'+str(i)+'/'+INLET['group'])
        c_inlet = np.array(g_inlet[INLET['point']])
        A_inlet = sv.calc_area_3d(c_inlet)
        flow_inlet = INLET['flow_val']*1.0/A_inlet
        l_inlet = "0 {}\n1 {}".format(flow_inlet, flow_inlet)
        d = {"area":A_inlet, "flow":flow_inlet}
        io.write_json(d,mo+'/'+str(i)+'/inlet.json')

        group_areas = {}
        for group,group_fn in zip(GROUPS,GROUPS_FILES):
            g_fn = mo+'/'+str(i)+'/'+group_fn
            g_j  = io.load_json(g_fn)
            c_id = list(g_j.keys())[-1]
            c    = np.array(g_j[c_id])
            A    = sv.calc_area_3d(c)
            group_areas[group]=A

        io.write_json(group_areas,mo+'/'+str(i)+'/'+'areas.json')

        total_area = 0
        for k,v in group_areas.items(): total_area+=v
        ratios = {}
        for k,v in group_areas.items(): ratios[k] = 1.0/v
        total_ratios = 0
        for k,v in ratios.items(): total_ratios+=v
        for k in ratios: ratios[k] = ratios[k]/total_ratios
        io.write_json(ratios,mo+'/'+str(i)+'/'+'ratios.json')

        resistances = {}
#        R = TOTAL_R/np.prod([v for k,v in ratios.items()])
        R = TOTAL_R*len(GROUPS)
        for k,v in ratios.items(): resistances[k]=v*R
        io.write_json(resistances,mo+'/'+str(i)+'/'+'resistances.json')

        for me in MESH_DIRS:
            print(i,me)
            try:
                me_dir = g+'/'+me+'/'+str(i)

                os.system('rm -r {}'.format(me_dir+'/'+SIM_NAME))
                os.system('cp -r {} {}'.format(SIM_DIR,me_dir))

                # new_solv_file = me_dir+'/'+SIM_NAME+'/'+SOLVER_FILE
                #
                # with open(new_solv_file,'w') as f:
                #     for l in SOLVER_LINES:
                #         if '$' in l:
                #             for k,v in resistances.items():
                #                 key = '$'+k+'$'
                #                 l = l.replace(key,str(resistances[k]))
                #         f.write(l)

                new_flow_file = me_dir+'/'+SIM_NAME+'/'+INLET['file']
                with open(new_flow_file,'w') as f:
                    f.write(l_inlet)
            except:
                print("failed")
