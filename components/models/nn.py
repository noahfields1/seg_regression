from base.model import AbstractModel
import numpy as np
import tensorflow as tf
import modules.layers as tf_util
import matplotlib.pyplot as plt
import modules.vessel_regression as vr

def get_batch(X,Y, batch_size=16):
    ids = np.random.choice(X.shape[0], size=batch_size)

    x   = np.array([X[i] for i in ids])
    y   = np.array([Y[i] for i in ids])

    return x,y

class Model(AbstractModel):
    def setup(self):
        self.build_model()
        self.configure_trainer()
        self.finalize()

    def train_step(self,x,y):
        self.global_step = self.global_step+1

        if np.sum(np.isnan(x)) > 0: return
        if np.sum(np.isnan(y)) > 0: return

        self.sess.run(self.train_op,{self.x:x,self.y:y})

    def save(self, model_path=None):
        if model_path == None:
            model_path = self.config['MODEL_DIR']+'/'+self.config['MODEL_NAME']
        else:
            model_path = model_path + '/' + self.config['MODEL_NAME']
        self.saver.save(self.sess,model_path)

    def load(self, model_path=None):
        if model_path == None:
            model_path = self.config['MODEL_DIR']+'/'+self.config['MODEL_NAME']
        else:
            model_path = model_path+'/'+self.config['MODEL_NAME']
        self.saver.restore(self.sess, model_path)

    def predict(self,x):
        S = list(x.shape)
        if len(S) == 3:
            x_ = x.reshape([1]+S)
            return self._predict(x_)[0]
        else:
            out = []
            for i in range(S[0]):
                x_ = x[i].reshape([1]+S[1:4])
                y = self._predict(x_)[0].copy()
                out.append(y)
            return np.array(out)

    def calculate_loss(self,x,y):
        return self.sess.run(self.loss,{self.x:x,self.y:y})

    def build_model(self):
        raise RuntimeError("Abstract not implemented")

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
        #self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def configure_trainer(self):
        LEARNING_RATE = self.config["LEARNING_RATE"]
        self.global_step = tf.Variable(0, trainable=False)
        boundaries = [200, 4000, 6000]
        values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/100, LEARNING_RATE/1000]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

        self.opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        self.train_op = self.opt.minimize(self.loss)

    def train(self, X,Y):
        for i in range(self.config['TRAIN_STEPS']):
            x,y = get_batch(X,Y, self.config['BATCH_SIZE'])

            self.train_step(x,y)

            if i % self.config['LOG_STEP'] == 0:
                self.log(i,x,y)
                self.log(i,X[:4],Y[:4])
                self.save()

    def log(self,i,x,y):
        l = self.calculate_loss(x,y)
        yhat = self.predict(x)[0]

        print("{}: loss={}\n".format(i,l))
        print("yhat = {}".format(yhat))

        f = open(self.config["LOG_FILE"],"a+")
        f.write("{}: loss={}\n".format(i,l))
        f.write("{}: yhat={}\n".format(i,yhat))
        f.close()

        self.save()

        x_ = x[0,:,:,0]
        y_ = y[0]
        ctrue = vr.pred_to_contour(y_)
        cpred = vr.pred_to_contour(yhat)

        plt.figure()
        plt.imshow(x_,cmap='gray',extent=[-1, 1, 1, -1])
        plt.colorbar()
        plt.scatter(cpred[:,0], cpred[:,1], color='r', label='predicted',s=4)
        plt.scatter(ctrue[:,0], ctrue[:,1], color='y', label='true', s=4)
        plt.show()
        plt.close()

class I2INetReg(Model):
    def build_model(self):
        CROP_DIMS   = self.config['CROP_DIMS']
        C           = self.config['NUM_CHANNELS']
        LEAK        = self.config['LEAK']
        NUM_FILTERS = self.config['NUM_FILTERS']
        LAMBDA      = self.config['L2_REG']
        INIT        = self.config['INIT']
        if "INIT" in self.config:
            INIT = self.config['INIT']
            print(INIT)


        NUM_POINTS  = self.config['NUM_CONTOUR_POINTS']

        leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

        self.x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None,NUM_POINTS],dtype=tf.float32)

        self.yclass,self.yhat,_,_ = tf_util.I2INet(self.x,nfilters=NUM_FILTERS,
            activation=leaky_relu,init=INIT)

        o = leaky_relu(self.yhat)

        s = o.get_shape().as_list()

        o_vec = tf.reshape(o,shape=[-1,s[1]*s[2]*s[3]])

        for i in range(self.config['FC_LAYERS']-1):
            if "HIDDEN_SIZES" in self.config:
                h = self.config['HIDDEN_SIZES'][i]
            else:
                h = self.config['HIDDEN_SIZE']

            o_vec = tf_util.fullyConnected(o_vec, h,
                leaky_relu, std=INIT, scope='fc_'+str(i))

        self.yhat = tf_util.fullyConnected(o_vec, NUM_POINTS,
            tf.identity, std=INIT, scope='fc_final')

        self.build_loss()

        self.saver = tf.train.Saver()

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
       # self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def _predict(self,x):
        return self.sess.run(self.yhat,{self.x:x})

    def finalize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

class ResNetReg(Model):
    def build_model(self):
        CROP_DIMS   = self.config['CROP_DIMS']
        C           = self.config['NUM_CHANNELS']
        LEAK        = self.config['LEAK']
        LAMBDA      = self.config['L2_REG']
        INIT        = self.config['INIT']

        NLAYERS     = int(self.config['NLAYERS']/2)
        NFILTERS_SMALL = self.config['NFILTERS_SMALL']
        NFILTERS_LARGE = self.config['NFILTERS_LARGE']

        NUM_POINTS  = self.config['NUM_CONTOUR_POINTS']

        leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

        self.x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None,NUM_POINTS],dtype=tf.float32)

        self.yclass,self.yhat,_,_ = tf_util.resNet(self.x,
            nlayers_before=NLAYERS, nlayers_after=NLAYERS,
            nfilters=NFILTERS_SMALL, nfilters_large=NFILTERS_LARGE,
            output_filters=NFILTERS_LARGE, activation=leaky_relu, init=INIT)

        o = leaky_relu(self.yhat)

        s = o.get_shape().as_list()

        o_vec = tf.reshape(o,shape=[-1,s[1]*s[2]*s[3]])

        for i in range(self.config['FC_LAYERS']-1):
            if "HIDDEN_SIZES" in self.config:
                h = self.config['HIDDEN_SIZES'][i]
            else:
                h = self.config['HIDDEN_SIZE']

            o_vec = tf_util.fullyConnected(o_vec, h,
                leaky_relu, std=INIT, scope='fc_'+str(i))

        self.yhat = tf_util.fullyConnected(o_vec, NUM_POINTS,
            tf.identity, std=INIT, scope='fc_final')

        self.build_loss()

        self.saver = tf.train.Saver()

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
        #self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def _predict(self,x):
        return self.sess.run(self.yhat,{self.x:x})

    def finalize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def configure_trainer(self):
        LEARNING_RATE = self.config["LEARNING_RATE"]
        self.global_step = tf.Variable(0, trainable=False)
        boundaries = [2000, 3000, 4000, 6000, 9000]
        values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/10000, LEARNING_RATE/100000, LEARNING_RATE/1000000, LEARNING_RATE/10000000]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

        #self.opt = tf.train.AdamOptimizer(learning_rate)
        self.opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)

        self.gvs = self.opt.compute_gradients(self.loss)
        self.capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self.gvs]
        self.train_op = self.opt.apply_gradients(self.capped_gvs)

class ResNetRegMultiscale(Model):
    def build_model(self):
        CROP_DIMS   = self.config['CROP_DIMS']
        C           = self.config['NUM_CHANNELS']
        LEAK        = self.config['LEAK']
        LAMBDA      = self.config['L2_REG']
        INIT        = self.config['INIT']

        NLAYERS     = int(self.config['NLAYERS']/2)
        NFILTERS_SMALL = self.config['NFILTERS_SMALL']
        NFILTERS_LARGE = self.config['NFILTERS_LARGE']

        NUM_POINTS  = self.config['NUM_CONTOUR_POINTS']

        leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

        self.x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)

        if self.config['MULTI_TYPE'] == "POOL":
            self.x_1 = tf.nn.pool(self.x, [2,2], "MAX", "VALID", strides=[2,2])
            self.x_2 = tf.nn.pool(self.x_1, [2,2], "MAX", "VALID", strides=[2,2])
        elif self.config['MULTI_TYPE'] == "CROP":
            self.x_1 = tf.image.central_crop(self.x, central_fraction=0.5)
            self.x_2 = tf.image.central_crop(self.x_1, central_fraction=0.5)
        else:
            raise RuntimeError("Unrecognized multi type")

        self.y = tf.placeholder(shape=[None,NUM_POINTS],dtype=tf.float32)

        self.yclass,self.yhat,_,_ = tf_util.resNet(self.x,
            nlayers_before=NLAYERS, nlayers_after=NLAYERS,
            nfilters=NFILTERS_SMALL, nfilters_large=NFILTERS_LARGE,
            output_filters=NFILTERS_LARGE, activation=leaky_relu, init=INIT)

        self.yclass_1,self.yhat_1,_,_ = tf_util.resNet(self.x_1,
            nlayers_before=NLAYERS, nlayers_after=NLAYERS,
            nfilters=NFILTERS_SMALL, nfilters_large=NFILTERS_LARGE,
            output_filters=NFILTERS_LARGE, activation=leaky_relu, init=INIT,
            scope="resnet_1")

        self.yclass_2,self.yhat_2,_,_ = tf_util.resNet(self.x_2,
            nlayers_before=NLAYERS, nlayers_after=NLAYERS,
            nfilters=NFILTERS_SMALL, nfilters_large=NFILTERS_LARGE,
            output_filters=NFILTERS_LARGE, activation=leaky_relu, init=INIT,
            scope="resnet_2")


        o   = leaky_relu(self.yhat)
        o_1 = leaky_relu(self.yhat_1)
        o_2 = leaky_relu(self.yhat_2)

        s   = o.get_shape().as_list()
        s_1 = o_1.get_shape().as_list()
        s_2 = o_2.get_shape().as_list()

        o_vec   = tf.reshape(o,shape=[-1,s[1]*s[2]*s[3]])
        o_vec_1 = tf.reshape(o_1,shape=[-1,s_1[1]*s_1[2]*s_1[3]])
        o_vec_2 = tf.reshape(o_2,shape=[-1,s_2[1]*s_2[2]*s_2[3]])

        o = tf.concat([o_vec, o_vec_1, o_vec_2], axis=1)

        for i in range(self.config['FC_LAYERS']-1):
            if "HIDDEN_SIZES" in self.config:
                h = self.config['HIDDEN_SIZES'][i]
            else:
                h = self.config['HIDDEN_SIZE']

            o_vec = tf_util.fullyConnected(o_vec, h,
                leaky_relu, std=INIT, scope='fc_'+str(i))

        self.yhat = tf_util.fullyConnected(o_vec, NUM_POINTS,
            tf.identity, std=INIT, scope='fc_final')

        self.build_loss()

        self.saver = tf.train.Saver()

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
        #self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def _predict(self,x):
        return self.sess.run(self.yhat,{self.x:x})

    def finalize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def configure_trainer(self):
        LEARNING_RATE = self.config["LEARNING_RATE"]
        self.global_step = tf.Variable(0, trainable=False)
        boundaries = [2000, 3000, 4000, 6000, 9000]
        values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/10000, LEARNING_RATE/100000, LEARNING_RATE/1000000, LEARNING_RATE/10000000]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

        #self.opt = tf.train.AdamOptimizer(learning_rate)
        self.opt = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
        #
        # self.gvs = self.opt.compute_gradients(self.loss)
        # self.capped_gvs = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in self.gvs]
        # self.train_op = self.opt.apply_gradients(self.capped_gvs)
        self.train_op = self.opt.minimize(self.loss)

    def log(self,i,x,y):
        l = self.calculate_loss(x,y)
        yhat = self.predict(x)[0]

        x_1,x_2 = self.sess.run([self.x_1,self.x_2], {self.x:x})

        print("{}: loss={}\n".format(i,l))
        print("yhat = {}".format(yhat))

        f = open(self.config["LOG_FILE"],"a+")
        f.write("{}: loss={}\n".format(i,l))
        f.write("{}: yhat={}\n".format(i,yhat))
        f.close()

        self.save()

        x_ = x[0,:,:,0]
        y_ = y[0]
        ctrue = vr.pred_to_contour(y_)
        cpred = vr.pred_to_contour(yhat)

        plt.figure()
        plt.imshow(x_,cmap='gray',extent=[-1, 1, 1, -1])
        plt.colorbar()
        plt.scatter(cpred[:,0], cpred[:,1], color='r', label='predicted',s=4)
        plt.scatter(ctrue[:,0], ctrue[:,1], color='y', label='true', s=4)
        plt.show()
        plt.close()

        if self.config['MULTI_TYPE'] == "POOL":
            ext1 = [-1,1,1,-1]
            ext2 = [-1,1,1,-1]
        else:
            ext1 = [-0.5,0.5,0.5,-0.5]
            ext2 = [-0.25,0.25,0.25,-0.25]

        plt.figure()
        plt.imshow(x_1[0,:,:,0],cmap='gray',extent=ext1)
        plt.colorbar()
        plt.scatter(cpred[:,0], cpred[:,1], color='r', label='predicted',s=4)
        plt.scatter(ctrue[:,0], ctrue[:,1], color='y', label='true', s=4)
        plt.show()
        plt.close()

        plt.figure()
        plt.imshow(x_2[0,:,:,0],cmap='gray',extent=ext2)
        plt.colorbar()
        plt.scatter(cpred[:,0], cpred[:,1], color='r', label='predicted',s=4)
        plt.scatter(ctrue[:,0], ctrue[:,1], color='y', label='true', s=4)
        plt.show()
        plt.close()
