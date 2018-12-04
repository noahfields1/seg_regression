from base.model import AbstractModel
import numpy as np

def get_batch(X,Y, batch_size=16):
    ids = np.random.choice(X.shape[0])

    x   = np.array([X[i] for i in ids])
    y   = np.array([Y[i] for i in ids])

    return x,y

class BaseTFModel(AbstractModel):
    def train(X,Y):
        for i in range(self.config['TRAIN_STEPS']):
            x,y = get_batch(X,Y)

            self.train_step(x,y)

            if i % self.config['LOG_STEP'] == 0:
                self.log(i,x,y)
                self.save()
