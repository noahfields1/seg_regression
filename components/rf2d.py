import joblib
import numpy as np
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

from components.common import BaseExperiment
from base.model import AbstractModel

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

class RF2DExperiment(BaseExperiment):
    def train(self):
        self.model.train(self.Xnorm,self.C)
