import numpy as np
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
    Yc   = [d[2] for d in data]
    meta = [d[3] for d in data]

    points   = []
    contours = []
    for i,yc in tqdm(enumerate(Yc)):

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

    points   = np.array(points)
    contours = np.array(contours)

    return X,contours,points,meta
