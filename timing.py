import numpy as np
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import factories.model_factory as model_factory
from modules import io
from guppy import hpy; h=hpy()

cfg = io.load_yaml('./config/googlenet_c30_train300k_aug10_clean.yaml')
print(h.heap())
net = model_factory.get(cfg)
net.load()
print(h.heap())

a = np.random.randn(1,160, 160,1)
print('predicting')
t1 = time.time()

for i in range(100):
    b = net.predict(a)

t2 = time.time()
d = t2-t1

print("total time: {}".format(d))
print("time per run {}".format(d*1.0/100))
