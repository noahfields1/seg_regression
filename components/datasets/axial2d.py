import numpy as np
import random
from tqdm import tqdm
import skimage.filters as filters
import modules.dataset as dataset
from modules.io import load_yaml
import modules.vascular_data as sv
from base.dataset import AbstractDataset
EPS=1e-5

def read_T(id):
    meta_data = load_yaml(id)
    X         = np.load(meta_data['X'])
    Y         = np.load(meta_data['Y'])
    Yc        = np.load(meta_data['Yc'])
    return (X,Y,Yc,meta_data)

def radius_balance(X,Y,meta, r_thresh, Nsample):
    N = X.shape[0]
    radiuses = [m['radius']*m['spacing'] for m in meta]

    i_sm     = [i for i in range(N) if radiuses[i] <= r_thresh]
    i_lg     = [i for i in range(N) if radiuses[i] > r_thresh]

    index    = random.choices(i_sm,k=Nsample)+random.choices(i_lg,k=Nsample)

    X_ = np.array([X[i] for i in index])
    Y_ = np.array([Y[i] for i in index])
    m_ = [meta[i] for i in index]

    return X_,Y_,m_

def distance_contour(yc,cd, nc):
    c = sv.marchingSquares(yc, iso=0.5)
    c = sv.reorder_contour(c)

    c = (1.0*c-cd)/(cd)
    p = np.mean(c,axis=0)
    c_centered = c-p

    c_centered = c_centered[:,:2]
    p = p[:2]

    c_reorient = sv.interpContour(c_centered, num_pts=nc)

    c_dist = np.sqrt(np.sqrt(np.sum(c_reorient**2,axis=1)))

    return c_dist, p

def get_dataset(config, key="TRAIN"):
    """
    setup and return requested dataset
    args:
        config - dict   - must containt FILES_LIST
        key    - string - either TRAIN, VAL, or TEST
    """

    files = open(config['FILE_LIST']).readlines()
    files = [s.replace('\n','') for s in files]

    if key == "TRAIN":
        patterns = config['TRAIN_PATTERNS']
    elif key == "VAL":
        patterns = config['VAL_PATTERNS']
    elif key == "TEST":
        patterns = config['TEST_PATTERNS']
    else:
        raise RuntimeError("Unrecognized data key {}".format(key))

    files = [f for f in files if any([s in f for s in patterns])]

    data = [read_T(s) for s in files]

    meta = [d[3] for d in data]

    X    = np.array([d[0] for d in data])

    cr   = int(X.shape[1]/2)
    cc   = int(config['CENTER_DIMS']/2)
    cd   = int(config['CROP_DIMS']/2)

    Yc   = np.array([d[2] for d in data])

    X_center = []
    Y_center = []

    print("centering images")
    for i,yc in tqdm(enumerate(Yc)):
        try:
            contour = sv.marchingSquares(yc, iso=0.5)
            contour = sv.reorder_contour(contour)

            cx = np.mean(contour[:,0])
            cy = np.mean(contour[:,1])

            X_center.append( X[i,cy-cc:cy+cc, cx-cc:cx+cc] )
            Y_center.append( Yc[i,cy-cc:cy+cc, cx-cc:cx+cc] )
        except:
            print(meta[i])

    if config['BALANCE_RADIUS'] and key=='TRAIN':
        X_center,Y_center,meta = radius_balance(X_center,Y_center,meta,
        config['R_SMALL'], config['N_SAMPLE'])

    if "AUGMENT" in config:
        aug_x = []
        aug_y = []
        aug_m = []
        for k in config['AUGMENT_FACTOR']:
            for i in range(X.shape[0]):
                x = X_center[i]
                y = Y_center[i]

                x,y = sv.random_rotate((x,y))
                
                rpix = meta[i]['radius']
                lim  = int(np.sqrt(config['AUGMENT_R_SCALE']*rpix))
                x_shift = np.random.randint(-lim,lim)
                y_shift = np.random.randint(-lim,lim)

                aug_x.append( x[cc+y_shift-cd:cc+y_shift+cd, cc+x_shift-cd:cc+x_shift+cd] )
                aug_x.append( y[cc+y_shift-cd:cc+y_shift+cd, cc+x_shift-cd:cc+x_shift+cd] )
                aug_x.append( meta[i] )
        X = np.array(aug_x)
        Y = np.array(aug_y)
        meta = aug_m
    else:
        X = np.array(X_center)[:,cc-cd:cc+cd,cc-cd:cc+cd]
        Y = np.array(Y_center)[:,cc-cd:cc+cd,cc-cd:cc+cd]

    return X,Y,meta
