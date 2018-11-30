import joblib
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from experiments.components import AbstractExperiment, AbstractModel
from experiments.experiment import log_prediction

from modules.io import mkdir, write_json
from modules.vessel_regression import pred_to_contour
EPS = 1e-5

class RFModel(AbstractModel):
    def setup(self):
        n_estimators = self.config['N_ESTIMATORS']
        self.rf      = RandomForestRegressor(n_estimators=n_estimators)
        self.name = self.config['NAME']
    def predict(self, x):
        """
        x - array [HxW] or [NxHxW]
        """
        S = x.shape
        if len(S) == 2:
            return self._predict_single(x)
        else:
            return self._predict(x)

    def _predict_single(self, x):
        S  = x.shape
        x_ = x.reshape([1,s[0]*s[1]])
        p  = self.rf.predict(x_)
        return p[0]

    def _predict(self, X):
        S  = X.shape
        X_ = X.reshape([-1,S[1]*S[2]])
        return self.rf.predict(X_)

    def train(self, X, Y):
        S  = X.shape
        X_ = X.reshape([-1,S[1]*S[2]])
        self.rf.fit(X_,Y)

    def save(self, path):
        joblib.dump(self.rf,path+'/{}.joblib'.format(self.name))

    def load(self, path):
        self.rf = joblib.load(path+'/{}.joblib'.format(self.name))

class RF2DExperiment(AbstractExperiment):
    def __init__(self, config):
        self.config = config
        self.setup()

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
        self.model = RFModel(self.config)

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

    def train(self):
        self.model.train(self.Xnorm,self.C)
    def save(self):
        self.model.save(self.model_dir)
    def load(self):
        self.model.load(self.model_dir)
