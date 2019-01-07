import components.common as common

def get(config, key="TRAIN"):
    if "POST_PROCESSOR" in config:
        pp = config['POST_PROCESSOR']

        if pp == "EDGE":
            return common.EdgePostProcessor(config)
    return BasePostProcessor(config)
