
def get(config, key="TRAIN"):
    if not "DATASET" in config:
        raise RuntimeError("No DATASET key specified in config")

    dset = config['DATASET']

    if dset == "axial2d":
        return axial2d.get_dataset(config, key)
    else:
        raise RuntimeError("Unrecognized dataset {}".format(dset))
