import modules.dataset as dataset

def get_dataset(config, key="TRAIN"):
    """
    setup and return requested dataset
    args:
        config - dict   - must containt FILES_LIST
        key    - string - either TRAIN, VAL, or TEST
    """

    files = open(config['FILES_LIST']).readlines()
    files = [s.replace('\n','') for s in files]

    #TODO: other stuff, get data, return arrays or dataset obj
