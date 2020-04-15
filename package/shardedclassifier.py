import copy
import numpy as np
from collections import Counter
from package import ensembleselection


#   VanillaShardedClassifier:
#       - Prediction - vanilla implementation: taking simple majority vote
#       - Unlearning - vanilla implementation: remove point and refit model 
#       (in case of auto sklearn refit causes rediscovery with bayesian optimizer)
#   Expects:
#   1. Training class labels to be in range 0..n
#   2. Unlearning calls should provide index of point to be unlearnt in the original training data set passed to fit()

class VanillaShardedClassifier:

    def __init__(self, num_shards=1, ml_algorithm=None):
        # Sharded Classifier characteristics:
        #   num_shards = number of shards to create
        #   ml_algorithm = ml algorithm with which each shard is to be trained
        self.num_shards = num_shards
        self.ml_algorithm = ml_algorithm

        # Training data
        self.X_train = None
        self.y_train = None

        # Book-keeping
        #   shard_data_dict: dictionary {shard_id, <list>}
        #       - mapping of training points in each shard
        #   shard_model_dict: dictionary {shard_id, model}
        #       - mapping of fitted model for each shard
        #   data_to_shard_dict: dictionary {training point idx, shard id}
        #       - reverse mapping of each point in training data to shard in which it lies
        #       - this is currently a hack to find shard during unlearning in O(1)
        #       - ideally this should be replaced by binary search
        self.shard_data_dict = {}
        self.shard_model_dict = {}
        self.data_to_shard_dict = {}

        # Default prediction in case of 0 points in training set
        self.default_class = None

    def fit(self, X, y):
        self.X_train = copy.deepcopy(X)
        self.y_train = copy.deepcopy(y)
        self.default_class = Counter(y).most_common(1)[0][0]
        self.initialize_bookkeeping_dicts()
        for shard_num in self.shard_data_dict:
            self.shard_model_dict[shard_num] = self.fit_shard(self.X_train[self.shard_data_dict[shard_num]],
                                                              self.y_train[self.shard_data_dict[shard_num]])

    # Prediction - vanilla implementation: taking simple majority vote
    def predict(self, X):
        predictions = []
        for m in self.shard_model_dict:
            predictions.append(self.shard_model_dict[m].predict(X))
        predictions = np.asarray(predictions)
        ret_predictions = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(len(X))]
        return ret_predictions

    # Unlearning - vanilla implementation: remove point and refit model
    #               (in case of auto sklearn refit causes rediscovery with bayesian optimizer)
    def unlearn(self, X_y_ids):
        shard_num = self.getShardNum(X_y_ids)
        for i in range(len(X_y_ids)):
            self.shard_data_dict[shard_num[i]].remove(X_y_ids[i])
        for shard_i in list(set(shard_num)):
            self.shard_model_dict[shard_i] = self.fit_shard(self.X_train[self.shard_data_dict[shard_i]],
                                                            self.y_train[self.shard_data_dict[shard_i]])

    def fit_shard(self, X, y):
        if len(X) is 0:
            return self.DummyClassifier(prediction=self.default_class)
        elif len(Counter(y).keys()) is 1:
            return self.DummyClassifier(prediction=y[0])
        else:
            return self.ml_algorithm.fit(X, y)

    #   Creates class-balanced separations of training data and assigns to each shard
    def initialize_bookkeeping_dicts(self):
        y = self.y_train
        manager = [0] * len(Counter(y).keys())
        self.shard_data_dict = {sh_num: [] for sh_num in range(self.num_shards)}
        for it in range(len(y)):
            self.shard_data_dict[manager[y[it]]].append(it)
            self.data_to_shard_dict[it] = manager[y[it]]
            manager[y[it]] = (manager[y[it]] + 1) % self.num_shards
        self.shard_model_dict = {sh_num: None for sh_num in range(self.num_shards)}

    def getShardNum(self, idx):
        return [self.data_to_shard_dict[id_i] for id_i in idx]

    class DummyClassifier:
        prediction = 0

        def __init__(self, prediction):
            self.prediction = prediction

        def predict(self, X):
            return [self.prediction] * len(X)


#   EnsembleShardedClassifier: extends VanillaShardedClassifier:
#       - Fits an ensemble of independent shard models using ensembleselction.EnsembleSelectionClassifier
#       - Prediction using ensemble model
#       - Unlearning followed by creating a new ensemble using retrained shard models
class EnsembleShardedClassifier(VanillaShardedClassifier):
    ensembleModel = None

    def fit(self, X, y):
        super().fit(X, y)
        self.ensembleModel = ensembleselection.EnsembleSelectionClassifier().getEnsemble(
            list(self.shard_model_dict.values()))

    def predict(self, X):
        return self.ensembleModel.predict(X)

    def unlearn(self, X_y_ids):
        super().unlearn(X_y_ids)
        self.ensembleModel = ensembleselection.EnsembleSelectionClassifier().getEnsemble(
            list(self.shard_model_dict.values()))
