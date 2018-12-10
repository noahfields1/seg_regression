from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from base.model import AbstractModel

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

    def save(self, path=None):
        if path == None:
            path = os.path.join(self.config['RESULTS_DIR'], self.name, 'model')
        joblib.dump(self.rf,path+'/{}.joblib'.format(self.name))

    def load(self, path=None):
        if path == None:
            path = os.path.join(self.config['RESULTS_DIR'], self.name, 'model')
        self.rf = joblib.load(path+'/{}.joblib'.format(self.name))
