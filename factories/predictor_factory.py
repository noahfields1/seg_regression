import components.common as common

def get_predictor(config):
    if not "PREDICTOR" in config:
        raise RuntimeError("PREDICTOR key missing in config")

    predictor = config['PREDICTOR']

    if predictor == "rf2d":
        return common.BasePredictor(config)
    else:
        raise RuntimeError("Unrecognized predictor {}".format(predictor))
