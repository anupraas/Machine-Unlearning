class EnsembleSelectionClassifier:

    def __init__(self):
        pass

    # Expects a list of models in "models" and a validation set in "X", "y".
    # Returns an ensemble
    def getEnsemble(self, models, X, y):
        return DummyClassifier()


class DummyClassifier:

    def __init__(self, prediction=0):
        self.prediction = prediction

    def predict(self, X):
        return [self.prediction] * len(X)
