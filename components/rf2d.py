from components.common import BaseExperiment

class RF2DExperiment(BaseExperiment):
    def train(self):
        self.model.train(self.Xnorm,self.C)
