from components.common import BasePostProcessor

def get(config, key="TRAIN"):
    return BasePostProcessor(config)
