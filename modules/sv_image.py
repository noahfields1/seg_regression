import modules.vascular_data as sv

class Image(object):
    def __init__(self, filename):
        self.image = sv.read_mha(filename)

    def get_reslice(self, p, n, v, spacing):
        return sv.getImageReslice(self.image, p, n, v, spacing, asnumpy=True)
