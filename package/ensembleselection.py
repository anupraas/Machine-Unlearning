from mlxtend.classifier import EnsembleVoteClassifier
from sklearn import metrics
import copy


class EnsembleSelectionClassifier:

    def __init__(self, epsilon=0.1, maxIter=10):
        self.epsilon = epsilon
        self.max_iter = maxIter

    # Expects a list of "wrapped" models in "models" and a validation set in "X", "y".
    # Returns an ensemble
    def getEnsemble(self, models, X, y, ret_weights=False):
        cur_weights = [0] * len(models)
        cur_accuracy = 0
        it = 0
        while True:
            candidates = []
            it += 1
            for i in range(len(models)):
                candidate_weights = copy.deepcopy(cur_weights)
                candidate_weights[i] += 1
                candidate_ensemble = EnsembleVoteClassifier(clfs=models, voting='soft', weights=candidate_weights,
                                                            refit=False)
                # Dummy call to fit to prevent errors
                candidate_ensemble.fit(X, y)
                candidate_delta_accuracy = metrics.accuracy_score(y, candidate_ensemble.predict(X)) - cur_accuracy
                candidates.append((candidate_delta_accuracy, i, candidate_ensemble))
            candidates.sort(reverse=True)
            if candidates[0][0] < self.epsilon or it > self.max_iter:
                break
            cur_weights[candidates[0][1]] += 1
            cur_accuracy += candidates[0][0]
            cur_ensemble = candidates[0][2]
        if ret_weights:
            return cur_ensemble, cur_weights
        return cur_ensemble
