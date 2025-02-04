#import components.models.rf as rf
import components.models.nn as nn
from modules import io

def get(config):
    if not "MODEL" in config:
        raise RuntimeError("MODEL key missing from config")

    mod = config["MODEL"]

    if mod == "rf2d":
        return rf.RFModel(config)
    elif mod == 'gb':
        return rf.GBModel(config)
    elif mod == "I2INetReg":
        return nn.I2INetReg(config)
    elif mod == "ResNetReg":
        return nn.ResNetReg(config)
    elif mod == "ResNetRegMultiscale":
        return nn.ResNetRegMultiscale(config)
    elif mod == "ConvNet":
        return nn.ConvNet(config)
    elif mod == "FcNet":
        return nn.FcNet(config)
    elif mod == "GoogleNet":
        return nn.GoogleNet(config)
    elif mod == "GoogleNetUQ":
        return nn.GoogleNetUQ(config)
    elif mod == "UNet":
        return nn.UNet(config)
    elif mod == "UNetUQ":
        return nn.UNetUQ(config)
    elif mod == "pretrained":
        return nn.PretrainedNet(config)

    elif mod == "edge_fit":
        c = './config/'+config['MODEL_YAML']
        y = io.load_yaml(c)
        return get(y)
    else:
        raise RuntimeError("Unrecognized model type {}".format(mod))
