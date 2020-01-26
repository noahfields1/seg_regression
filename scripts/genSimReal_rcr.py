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

RCR_FILE  = cfg['rcr_file']
RCR_LINES = open(SIM_DIR+'/'+RCR_FILE,'r').readlines()

C = cfg["capacitance"]

for g,mo in zip(GEN_DIRS,MODEL_DIRS):
    for i in range(NUM_MODELS):

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
        for k,v in group_areas.items(): ratios[k] = total_area*1.0/v

        io.write_json(ratios,mo+'/'+str(i)+'/'+'ratios.json')

        resistances = {}
#        R = TOTAL_R/np.prod([v for k,v in ratios.items()])
        R = TOTAL_R
        for k,v in ratios.items(): resistances[k]=v*R
        io.write_json(resistances,mo+'/'+str(i)+'/'+'resistances_rcr.json')


        # for k,v in group_areas.items(): ratios[k] = v
        # total_ratios = 0
        # for k,v in ratios.items(): total_ratios+=v
        # for k in ratios: ratios[k] = ratios[k]/total_ratios*len(GROUPS)

        capacitances = {}
        for k,v in ratios.items(): capacitances[k]=(1.0/v)*C
        io.write_json(capacitances,mo+'/'+str(i)+'/'+'capacitances.json')

        for me in MESH_DIRS:
            print(i,me)
            try:
                me_dir = g+'/'+me+'/'+str(i)

                os.system('rm -r {}'.format(me_dir+'/'+SIM_NAME))
                os.system('cp -r {} {}'.format(SIM_DIR,me_dir))

                new_solv_file = me_dir+'/'+SIM_NAME+'/'+RCR_FILE

                with open(new_solv_file,'w') as f:
                    for l in RCR_LINES:
                        lk = l.replace('\n','')
                        if lk in resistances:
                            r = resistances[lk]
                            c = capacitances[lk]

                            f.write(str(r*0.09)+'\n')
                            f.write(str(c)+'\n')
                            f.write(str(r*0.91)+'\n')

                        else:
                            f.write(l)

            except:
                print("failed")
