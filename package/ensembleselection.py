class EnsembleSelectionClassifier:

    def __init__(self):
        pass

    def getEnsemble(self, models, X, y):
        return DummyClassifier()


class DummyClassifier:

    def __init__(self, prediction=0):
        self.prediction = prediction

    def predict(self, X):
        return [self.prediction] * len(X)
