print("imported svWrapper.py")
import os

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
    def segment(self, point_string):
        print("test: point_string {}".format(point_string))
        return "test: output of segment"
