import numpy as np
import random
from tqdm import tqdm
import modules.dataset as dataset
from modules.io import load_yaml
import modules.vascular_data as sv

def read_T(id):
    meta_data = load_yaml(id)
    X         = np.load(meta_data['X'])
    Y         = np.load(meta_data['Y'])
    Yc        = np.load(meta_data['Yc'])
    return (X,Y,Yc,meta_data)

def radius_balance(X,c,p,meta, r_thresh, Nsample):
    N = X.shape[0]
    radiuses = [m['radius']*m['spacing'] for m in meta]

    i_sm     = [i for i in range(N) if radiuses[i] <= r_thresh]
    i_lg     = [i for i in range(N) if radiuses[i] > r_thresh]

    index    = random.choices(i_sm,k=Nsample)+random.choices(i_lg,k=Nsample)

    X_ = np.array([X[i] for i in index])
    c_ = np.array([c[i] for i in index])
    p_ = np.array([p[i] for i in index])
    m_ = [meta[i] for i in index]

    return X_,c_,p_,m_
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

    X    = np.array([d[0] for d in data])

    c    = int(X.shape[1]/2)
    cd   = int(config['CROP_DIMS']/2)

    X    = X[:,c-cd:c+cd,c-cd:c+cd]

    Yc   = np.array([d[2] for d in data])
    Yc   = Yc[:,c-cd:c+cd,c-cd:c+cd]

    meta = [d[3] for d in data]

    points   = []
    contours = []
    for i,yc in tqdm(enumerate(Yc)):
        try:
            c = sv.marchingSquares(yc, iso=0.5)
            c = sv.reorder_contour(c)

            H = yc.shape[1]
            c = (1.0*c-H/2)/(H/2)
            p = np.mean(c,axis=0)
            c_centered = c-p

            c_centered = c_centered[:,:2]
            p = p[:2]

            c_reorient = sv.interpContour(c_centered,
                num_pts=config['NUM_CONTOUR_POINTS'])

            c_dist = np.sqrt(np.sqrt(np.sum(c_reorient**2,axis=1)))

            p = (p+1)/2
            points.append(p)
            contours.append(c_dist)
        except:
            print(meta[i])

    points   = np.array(points)
    contours = np.array(contours)

    if config['BALANCE_RADIUS'] and key=='TRAIN':
        X,contours,points,meta = radius_balance(X,contours,points,meta,
            config['R_SMALL'], config['N_SAMPLE'])


    return X,contours,points,meta
