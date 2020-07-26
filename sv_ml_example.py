import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

SV_PATH       = '/home/gabriel/projects/SV/fork/SimVascular/Python/site-packages'
sys.path.append(SV_PATH)

from sv_ml import sv_wrapper

NET_FN = "googlenet_c30_train300k_aug10_clean"
IMAGE_FN = "/home/gabriel/projects/SV/svprojects/test/Images/0110.vti"

sw = sv_wrapper.SVWrapper(NET_FN)

sw.set_image(IMAGE_FN)

import json

d = {
    "p":[0.0,0.0,0.0],
    "n":[1.0,0.0,0.0],
    "tx":[0.0,1.0,0.0]
}

d_s = json.dumps(d)

ctr_pts_s = sw.segment(d_s)

ctr_pts = json.loads(ctr_pts_s)

print(type(ctr_pts))
print(ctr_pts.keys())
print(ctr_pts['points'])
