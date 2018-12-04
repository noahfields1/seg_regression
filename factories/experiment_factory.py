import components.rf2d as rf2d

def get_experiment(config):
    if not "EXPERIMENT" in config:
        raise RuntimeError("EXPERIMENT key missing in config")

    exp = config['EXPERIMENT']

    if exp == "rf2d":
        return rf2d.RF2DExperiment(config)
    else:
        raise RuntimeError("Unrecognized experiment {}".format(exp))
