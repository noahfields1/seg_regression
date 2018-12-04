from base.model import AbstractModel
import numpy as np
import tensorflow as tf
import modules.layers as tf_util

def get_batch(X,Y, batch_size=16):
    ids = np.random.choice(X.shape[0])

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

        self.sess.run(self.train,{self.x:x,self.y:y})

    def save(self):
        model_dir  = self.case_config['MODEL_DIR']
        model_name = self.case_config['MODEL_NAME']
        self.saver.save(
            self.sess,model_dir+'/{}'.format(model_name))

    def load(self, model_path=None):
        if model_path == None:
            model_dir  = self.case_config['MODEL_DIR']
            model_name = self.case_config['MODEL_NAME']
            model_path = model_dir + '/' + model_name
        self.saver.restore(self.sess, model_path)

    def predict(self,x):
        return self.sess.run(self.yclass,{self.x:x})

    def calculate_loss(self,x,y):
        return self.sess.run(self.loss,{self.x:x,self.y:y})

    def build_model(self):
        raise RuntimeError("Abstract not implemented")

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
        self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def configure_trainer(self):
        LEARNING_RATE = self.global_config["LEARNING_RATE"]
        self.global_step = tf.Variable(0, trainable=False)
        boundaries = [10000, 20000, 25000]
        values = [LEARNING_RATE, LEARNING_RATE/10, LEARNING_RATE/100, LEARNING_RATE/1000]
        learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, values)

        self.opt = tf.train.AdamOptimizer(learning_rate)
        self.train = self.opt.minimize(self.loss)

    def train(X,Y):
        for i in range(self.config['TRAIN_STEPS']):
            x,y = get_batch(X,Y)

            self.train_step(x,y)

            if i % self.config['LOG_STEP'] == 0:
                self.log(i,x,y)
                self.save()

    def log(self,i,x,y):
        l = self.calculate_loss(x,y)

        f = open(self.config["LOG_FILE"],"a+")
        f.write("{}: loss={}\n".format(i,l))
        f.close()

class I2INetReg(Model):
    def build_model(self):
        CROP_DIMS   = self.case_config['CROP_DIMS']
        C           = self.case_config['NUM_CHANNELS']
        LEAK        = self.global_config['LEAK']
        NUM_FILTERS = self.global_config['NUM_FILTERS']
        LAMBDA      = self.global_config['L2_REG']
        INIT        = self.global_config['INIT']
        if "INIT" in self.case_config:
            INIT = self.case_config['INIT']
            print(INIT)


        NUM_POINTS  = self.global_config['NUM_CONTOUR_POINTS']+2

        leaky_relu = tf.contrib.keras.layers.LeakyReLU(LEAK)

        self.x = tf.placeholder(shape=[None,CROP_DIMS,CROP_DIMS,C],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None,NUM_POINTS],dtype=tf.float32)

        self.yclass,self.yhat,_,_ = tf_util.I2INet(self.x,nfilters=NUM_FILTERS,
            activation=leaky_relu,init=INIT)

        o = leaky_relu(self.yhat)

        s = o.get_shape().as_list()

        o_vec = tf.reshape(o,shape=[-1,s[1]*s[2]*s[3]])

        for i in range(self.case_config['FC_LAYERS']-1):
            if "HIDDEN_SIZES" in self.case_config:
                h = self.case_config['HIDDEN_SIZES'][i]
            else:
                h = self.case_config['HIDDEN_SIZE']

            o_vec = tf_util.fullyConnected(o_vec, h,
                leaky_relu, std=INIT, scope='fc_'+str(i))

        self.yhat = tf_util.fullyConnected(o_vec, NUM_POINTS,
            tf.nn.sigmoid, std=INIT, scope='fc_final')

        self.build_loss()

        self.saver = tf.train.Saver()

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.square(self.y-self.yhat))
        self.loss += tf.reduce_mean(tf.abs(self.y-self.yhat))

    def predict(self,xb):
        return self.sess.run(self.yhat,{self.x:xb})

    def finalize(self):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
