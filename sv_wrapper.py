print("imported svWrapper.py")

import os
from modules import io
from modules import sv_image

import factories.model_factory as model_factory
import factories.preprocessor_factory as preprocessor_factory
import factories.postprocessor_factory as postprocessor_factory

print("imported factories")

SRC_DIR    = os.path.dirname(os.path.realpath(__file__))
CONFIG_DIR = os.path.join(SRC_DIR,"config")

class SVWrapper(object):
    def __init__(self, network_type):
        print("SVWrapper init, {}".format(network_type))
        self.cfg_fn = os.path.join(CONFIG_DIR,"{}.yaml".format(network_type))
        print(self.cfg_fn)

        if not os.path.isfile(self.cfg_fn):
            raise RuntimeError("network type does not exist {}".format(self.cfg_fn))

        self.cfg = io.load_yaml(self.cfg_fn)

        self.cfg['MODEL_DIR'] = self.cfg['MODEL_DIR'].replace('./results',
            '{}/results'.format(SRC_DIR))

        print("Model dir ", self.cfg['MODEL_DIR'])

        self.model = model_factory.get(self.cfg)

        print("loading model")
        self.model.load()
        print("model loaded")

    def set_image(self, image_fn):
        print("setting image {}".format(image_fn))
        if not os.path.isfile(image_fn):
            raise RuntimeError("image file not found {}".format(image_fn))

        self.image = sv_image(image_fn)
        print("successfully set image")
        self.image.set_reslice_ext(self.cfg['CROP_DIMS'])
        self.image.set_spacing(self.cfg['SPACING'])

        print("successfully set image")
        return "ok"

    def segment(self, point_string):
        print("test: point_string {}".format(point_string))
        return "test: output of segment"
