import numpy as np
import matplotlib.pyplot as plt

X = np.zeros((3,3))

px = np.random.randint(3)
py = np.random.randint(3)

w = np.random.randint(1,3-px+1)
h = np.random.randint(1,3-py+1)

X[py:py+h,px:px+w] = 1.0

#calc w
sw = np.sum(X,axis=1)
what = np.amax(sw)*1.0/3

#calc h
sh = np.sum(X,axis=0)
hhat = np.amax(sh)*1.0/3

#calc px
col_sum = np.sum(X,axis=0)
dx = np.diff(col_sum)
thresh_x = dx.copy()
thresh_x[thresh_x>0] = 1
thresh_x[thresh_x<0] = 0

theta_x = np.array([1.0/3, 2.0/3])

pxhat = np.sum(theta_x*thresh_x)

#calc px
row_sum = np.sum(X,axis=1)
dy = np.diff(row_sum)
thresh_y = dy.copy()
thresh_y[thresh_y>0] = 1
thresh_y[thresh_y<0] = 0

theta_y = np.array([1.0/3, 2.0/3])

pyhat = np.sum(theta_y*thresh_y)

xhat = np.array([pxhat, pxhat+what, pxhat+what, pxhat])
yhat = np.array([pyhat, pyhat, pyhat+hhat, pyhat+hhat])

print(pxhat,pyhat,what,hhat)

plt.figure()
plt.imshow(X, extent=[0, 1, 1, 0], cmap='gray')
plt.colorbar()
plt.scatter(xhat,yhat, color='r')
plt.show()
