import joblib
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from modules.io import mkdir, write_json
from modules.vessel_regression import pred_to_contour

from base.experiment import AbstractExperiment

def log_prediction(yhat,x,c,p,meta,path):
    cpred = pred_to_contour(yhat)
    ctrue = pred_to_contour(c)

    new_meta = {}
    for k in meta: new_meta[k] = meta[k]

    new_meta['center']   = p.tolist()
    new_meta['yhat_raw'] = yhat.tolist()
    new_meta['c_raw']    = c.tolist()

    new_meta['yhat_centered'] = cpred.tolist()
    new_meta['c_centered']    = ctrue.tolist()

    cpred_pos = cpred+p
    ctrue_pos = ctrue+p

    new_meta['yhat_pos'] = cpred_pos.tolist()
    new_meta['c_pos']    = ctrue_pos.tolist()

    name = meta['image']+'.'+meta['path_name']+'.'+str(meta['point'])

    write_json(new_meta, path+'/predictions/{}.json'.format(name))

    plt.figure()
    plt.imshow(x,cmap='gray',extent=[-1,1,1,-1])
    plt.colorbar()
    plt.scatter(cpred_pos[:,0], cpred_pos[:,1], color='r', label='predicted',s=4)
    plt.scatter(ctrue_pos[:,0], ctrue_pos[:,1], color='y', label='true', s=4)
    plt.legend()
    plt.savefig(path+'/images/{}.png'.format(name),dpi=200)
    plt.close()

class BaseExperiment(AbstractExperiment):
    def setup_directories(self):
        mkdir(self.root)
        mkdir(self.log_dir)
        mkdir(self.model_dir)
        mkdir(self.val_dir)
        mkdir(self.val_image_dir)
        mkdir(self.val_pred_dir)
        mkdir(self.test_dir)
        mkdir(self.test_image_dir)
        mkdir(self.test_pred_dir)

    def set_data(self, data, data_key):
        """
        data is a tuple (X,C,points,meta)
        """
        self.X      = data[0]
        self.C      = data[1]
        self.points = data[2]
        self.meta   = data[3]
        self.data_key = data_key

        #normalize X
        print("normalizing data")
        mu         = 1.0*np.mean(self.X,axis=(1,2), keepdims=True)
        sig        = 1.0*np.std(self.X,axis=(1,2), keepdims=True)+EPS
        self.Xnorm = (self.X-mu)/sig

    def setup(self):
        res_dir = self.config['RESULTS_DIR']
        name    = self.config['NAME']

        self.root           = os.path.join(res_dir,name)
        self.log_dir        = os.path.join(self.root,'log')
        self.model_dir      = os.path.join(self.root,'model')
        self.val_dir        = os.path.join(self.root,'val')
        self.val_image_dir  = os.path.join(self.root,'val','images')
        self.val_pred_dir   = os.path.join(self.root,'val','predictions')
        self.test_dir       = os.path.join(self.root,'test')
        self.test_image_dir = os.path.join(self.root,'test','images')
        self.test_pred_dir  = os.path.join(self.root,'test','predictions')

    def predict(self):
        predictions = self.model.predict(self.Xnorm)

        if self.data_key == "VAL":
            path = self.val_dir
        elif self.data_key == "TEST":
            path = self.test_dir

        for i in tqdm(range(predictions.shape[0])):
            x = self.Xnorm[i]
            c = self.C[i]
            p = self.points[i]
            meta = self.meta[i]
            yhat = predictions[i]

            log_prediction(yhat,x,c,p,meta,path)

    def save(self):
        self.model.save(self.model_dir)
    def load(self):
        self.model.load(self.model_dir)
