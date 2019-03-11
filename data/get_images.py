import os
import sys
sys.path.append(os.path.abspath('..'))

from modules import io

yamls = os.listdir('./cases')
yamls = [os.path.abspath('./cases') + '/' + f for f in yamls]

for yaml in yamls:
    y = io.load_yaml(yaml)

    name = y['NAME']
    try:
        os.mkdir('./images/{}'.format(name))
    except:
        print("{} directory already exists".format(name))
        
    im = y['IMAGE']
    seg = y['SEGMENTATION']

    im_path  = './images/{}/{}.mha'.format(name,'image')
    seg_path = './images/{}/{}.mha'.format(name,'segmentation')
    os.system('cp {} {}'.format(im,im_path))
    os.system('cp {} {}'.format(seg,seg_path))
