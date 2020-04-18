from itertools import chain
import numpy as np


class VotingEnsemble:
    def __init__(self, classifiers: [], classes: [], weights: []):
        self.classifiers = classifiers
        self.classes = classes
        self.weights = weights
        unique = set(chain(*classes))
        self.unique_classes_to_index = {}
        self.unique_index_to_classes = {}
        index = 0
        for i in unique:
            self.unique_classes_to_index[i] = index
            self.unique_index_to_classes[index] = i
            index += 1

    def predict(self, X_test, normalize = False):
        batch_size = len(X_test)
        prediction_matrix = np.zeros((batch_size, len(self.unique_classes_to_index)))
        counter = np.zeros(len(self.unique_classes_to_index))

        for i in range(len(self.classifiers)):
            clf = self.classifiers[i]
            probabilities = clf.predict_proba(X_test)
            probabilities = probabilities * self.weights[i]
            clf_classes = self.classes[i]
            for idx in range(len(clf_classes)):
                cls = clf_classes[idx]
                cls_idx = self.unique_classes_to_index[cls]
                prediction_matrix[:, cls_idx] += probabilities[:, idx]
                counter[cls_idx] += 1

        if normalize:
            prediction_matrix = prediction_matrix / counter
        predictions = np.argmax(prediction_matrix, axis= 1)
        predictions = [self.unique_index_to_classes[index] for index in predictions]

        return predictions

