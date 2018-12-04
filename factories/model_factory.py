import components.models.rf as rf
import components.models.nn as nn

def get_model(config):
    if not "MODEL" in config:
        raise RuntimeError("MODEL key missing from config")

    mod = config["MODEL"]

    if mod == "rf2d":
        return rf.RFModel(config)
    elif mod == "I2INetReg":
        return nn.I2INetReg(config)
    else:
        raise RuntimeError("Unrecognized model type {}".format(mod))
