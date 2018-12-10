import components.rf2d as rf2d

def get_predictor(config):
    if not "PREDICTOR" in config:
        raise RuntimeError("PREDICTOR key missing in config")

    predictor = config['PREDICTOR']

    if predictor == "rf2d":
        return rf2d.RF2DPredictor(config)
    else:
        raise RuntimeError("Unrecognized predictor {}".format(predictor))
