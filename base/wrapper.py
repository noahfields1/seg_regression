class AbstractWrapper(object):
    def __init__(self, model, postprocessor=None):
        self.model = model
        self.postprocessor = postprocessor

    def predict(self,x, preprocess=True):
        xnorm = x
        if preprocess:
            xnorm = self._preprocess(x)

        yhat  = self.model.predict(xnorm)

        if not self.postprocessor == None:
            yhat = self.postprocessor(x,yhat)

        c     = self._convert(yhat)

        return c

    def predict_raw(self, x, preprocess=True):
        xnorm = x
        if preprocess:
            xnorm = self._preprocess(x)

        return self.model.predict(xnorm)

    def _preprocess(x):
        pass

    def _convert(yhat):
        pass
