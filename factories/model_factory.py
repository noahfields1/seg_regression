import components.models.rf as rf

def get_model(config):
    if not "MODEL" in config:
        raise RuntimeError("MODEL key missing from config")

    mod = config["MODEL"]

    if mod == "rf2d":
        return rf.RFModel(config)
    else:
        raise RuntimeError("Unrecognized model type {}".format(mod))
