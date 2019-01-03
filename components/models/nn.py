from base.model import AbstractModel
import numpy as np
import tensorflow as tf
import modules.layers as tf_util

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
        self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def configure_trainer(self):
        LEARNING_RATE = self.config["LEARNING_RATE"]
        self.global_step = tf.Variable(0, trainable=False)
        boundaries = [10000, 20000, 25000]
        values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/100, LEARNING_RATE/1000]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.opt.minimize(self.loss)

    def train(self, X,Y):
        for i in range(self.config['TRAIN_STEPS']):
            x,y = get_batch(X,Y, self.config['BATCH_SIZE'])

            self.train_step(x,y)

            if i % self.config['LOG_STEP'] == 0:
                self.log(i,x,y)
                self.save()

    def log(self,i,x,y):
        l = self.calculate_loss(x,y)

        print("{}: loss={}\n".format(i,l))

        f = open(self.config["LOG_FILE"],"a+")
        f.write("{}: loss={}\n".format(i,l))
        f.close()

        self.save()

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
            tf.nn.sigmoid, std=INIT, scope='fc_final')

        self.build_loss()

        self.saver = tf.train.Saver()

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
        self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def _predict(self,x):
        return self.sess.run(self.yhat,{self.x:x})

    def finalize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
