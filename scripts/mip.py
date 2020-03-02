import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-i')
parser.add_argument('-o')

args = parser.parse_args()

img = sitk.ReadImage(args.i)

imgnp = sitk.GetArrayFromImage(img)

#plot maximum intensity projections
axes = [0,1,2]
for a in axes:
    i = np.amax(imgnp,axis=a)

    plt.figure()
    plt.imshow(i, cmap='gray')
    plt.savefig('{}/{}.png'.format(args.o,a),dpi=500)
    plt.close()
