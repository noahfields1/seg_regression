import components.rf2d as rf2d

def get_trainer(config):
    if not "TRAINER" in config:
        raise RuntimeError("TRAINER key missing in config")

    train = config['TRAINER']

    if train == "rf2d":
        return rf2d.RF2DTrainer(config)
    else:
        raise RuntimeError("Unrecognized trainer {}".format(train))
