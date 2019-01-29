import matplotlib.pyplot as plt
import sys
import os
import pygame
import numpy as np

sys.path.append(os.path.abspath('..'))

from components.datasets.axial2d import read_T, distance_contour
from modules.vessel_regression import pred_to_contour

CROP = 100
DIM  = 240
s = int(DIM/2-CROP/2)
e = int(DIM/2+CROP/2)

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
fig, ax = plt.subplots()

im = ax.imshow(Z,extent=[-1,1,1,-1], cmap='gray')
cbar = plt.colorbar(im)
cbar.ax.set_autoscale_on(True)
scat = ax.scatter([],[],color='r')

# pygame.init()

for i in range(track,N):
    cur_file = files[i]

    X,Y,Yc,meta = read_T(cur_file)

    X  = X[s:e,s:e]
    Yc = Yc[s:e,s:e]

    c,p = distance_contour(Yc,int(X.shape[0]/2),15)

    cp = pred_to_contour(c)

    im.set_data(X)
    cbar.set_clim(vmin=np.amin(X),vmax=np.amax(X))
    scat.set_offsets(cp)
    plt.pause(0.5)
    # while True:
    #     for event in pygame.event.get():
    #         if event.type == pygame.KEYUP:
    #             if event.key == pygame.K_w:
    #                 print("accepted")
    #                 break
    #             if event.key == pygame.K_q:
    #                 print("rejected")
    #                 break
