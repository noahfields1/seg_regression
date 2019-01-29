import matplotlib.pyplot as plt
import sys
import os
import pygame
from pygame.locals import *

import numpy as np

sys.path.append(os.path.abspath('..'))

from components.datasets.axial2d import read_T, distance_contour
from modules.vessel_regression import pred_to_contour

CROP = 140
CROP_SMALL = 40

DIM  = 240
ext_small = CROP_SMALL/CROP

s = int(DIM/2-CROP/2)
e = int(DIM/2+CROP/2)

ss = int(DIM/2-CROP_SMALL/2)
es = int(DIM/2+CROP_SMALL/2)

DATA_DIR  = "/media/marsdenlab/Data2/datasets/DeepLofting/files"

FILE_LIST = "/media/marsdenlab/Data2/datasets/DeepLofting/files/files.txt"
FILE_LIST_CLEAN = "/media/marsdenlab/Data2/datasets/DeepLofting/files/files_clean.txt"

files = open(FILE_LIST).readlines()
files = [s.replace('\n','') for s in files]

track = open("../other/track.txt").readlines()[0]
track = int(track)

f = open(FILE_LIST_CLEAN,'a')

N = len(files)

Z = np.zeros((240,240))
fig, (ax,ax_s) = plt.subplots(1,2)

im = ax.imshow(Z,extent=[-1,1,1,-1], cmap='gray',aspect='auto')
cbar = plt.colorbar(im)
cbar.ax.set_autoscale_on(True)
scat = ax.scatter([],[],color='r')

im_s = ax_s.imshow(Z,extent=[-ext_small,ext_small,ext_small,-ext_small], cmap='gray',aspect='auto')
cbar_s = plt.colorbar(im_s)
cbar_s.ax.set_autoscale_on(True)
scat_s = ax_s.scatter([],[],color='r')

pygame.init()
screen = pygame.display.set_mode((800, 600))

process_events = True

for i in range(track,N):
    process_events = True

    cur_file = files[i]

    X,Y,Yc,meta = read_T(cur_file)

    X_  = X[s:e,s:e]
    Yc_ = Yc[s:e,s:e]

    c,p = distance_contour(Yc_,int(X_.shape[0]/2),15)

    cp = pred_to_contour(c)

    im.set_data(X_)
    cbar.set_clim(vmin=np.amin(X_),vmax=np.amax(X_))
    scat.set_offsets(cp)

    im_s.set_data(X[ss:es,ss:es])
    cbar_s.set_clim(vmin=np.amin(X[ss:es,ss:es]),vmax=np.amax(X[ss:es,ss:es]))
    scat_s.set_offsets(cp)
    plt.pause(0.5)

    pygame.event.set_grab(True)
    while process_events:
        for event in pygame.event.get():
            if event.type == KEYUP:
                if event.key == pygame.K_w:
                    print("accepted")
                    f.write(cur_file+'\n')
                    process_events = False
                if event.key == pygame.K_q:
                    print("rejected")
                    process_events = False
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    f.close()
                    exit()

    g = open('../other/track.txt','w')
    g.write(str(i))
    g.close()
