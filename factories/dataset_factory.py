from components.datasets import axial2d, axial2d_seg, axial2d_point

def get(config, key="TRAIN"):
    if not "DATASET" in config:
        raise RuntimeError("No DATASET key specified in config")

    dset = config['DATASET']

    if dset == "axial2d":
        return axial2d.get_dataset(config, key)
    elif dset == 'axial2d_seg':
        return axial2d_seg.get_dataset(config, key)
    elif dset == 'axial2d_point':
        return axial2d_point.get_dataset(config, key)
    else:
        raise RuntimeError("Unrecognized dataset {}".format(dset))
