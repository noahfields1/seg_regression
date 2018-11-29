import numpy as np

def pred_to_contour(pred):
    #assume -1,1 extent
    Npoints  = len(pred)
    radiuses = pred**2
    angles   = np.linspace(-0.95,0.95, Npoints)*np.pi

    c = np.zeros((Npoints,2))

    c[:,0] = np.cos(angles)*radiuses
    c[:,1] = np.sin(angles)*radiuses

    return c
