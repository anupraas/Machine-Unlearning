import numpy as np
import copy


class modelWrapper:

    def __init__(self, model, num_classes, model_classes=None):
        self.model = copy.deepcopy(model)
        self.num_classes = num_classes
        if model_classes is None:
            self.model_classes = list(range(num_classes))
        else:
            self.model_classes = model_classes

    def fit(self, X, y):
        self.model.fit(X, y)
        self.model_classes = list(set(y))

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        proba = np.zeros(shape=(len(X), self.num_classes))
        mo_proba = self.model.predict_proba(X)
        for i in range(len(X)):
            for j in range(len(self.model_classes)):
                proba[i][self.model_classes[j]] = mo_proba[i][j]
        return proba
