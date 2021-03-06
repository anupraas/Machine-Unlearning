import copy
import numpy as np
from collections import Counter
from package import ensembleselection, modelwrapper as mw
import autosklearn.classification
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.dummy import DummyClassifier


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
        #   cur_train_ids: stored the current valid (unlearned) ids of the training set
        self.shard_data_dict = {}
        self.shard_model_dict = {}
        self.data_to_shard_dict = {}
        self.cur_train_ids = []

        # Default prediction in case of 0 points in training set
        self.default_class = None

        self.num_classes = None
        self.X_dummy = None
        self.y_dummy = None
        self.eclf = None

    def fit(self, X, y):
        self.X_train = copy.deepcopy(X)
        self.y_train = copy.deepcopy(y)
        self.default_class = Counter(y).most_common(1)[0][0]
        self.num_classes = len(set(self.y_train))
        self.X_dummy = np.zeros(shape=(self.num_classes, len(X[0])))
        self.y_dummy = np.asarray(list(range(self.num_classes)))
        self.num_shards = min(self.num_shards, Counter(y).most_common()[-1][1])
        self.initialize_bookkeeping_dicts()
        for shard_num in self.shard_model_dict:
            self.shard_model_dict[shard_num].fit(self.X_train[self.shard_data_dict[shard_num]],
                                                 self.y_train[self.shard_data_dict[shard_num]])
        self.eclf = EnsembleVoteClassifier(clfs=list(self.shard_model_dict.values()), weights=[1] * self.num_shards,
                                           voting='hard', refit=False)
        self.eclf.fit(self.X_dummy, self.y_dummy)

    # Prediction - vanilla implementation: taking simple majority vote
    def predict(self, X):
        return self.eclf.predict(X)

    # Unlearning - vanilla implementation: remove point and refit model
    #               (in case of auto sklearn refit causes rediscovery with bayesian optimizer)
    def unlearn(self, X_y_ids):
        shard_num = self.getShardNum(X_y_ids)
        for i in range(len(X_y_ids)):
            self.shard_data_dict[shard_num[i]].remove(X_y_ids[i])
            self.cur_train_ids.remove(X_y_ids[i])
        self.default_class = Counter(self.y_train[self.cur_train_ids]).most_common(1)[0][0]
        # Refitting shards after unlearning - vanilla implementation: call fit() for every shard's model
        for shard_i in list(set(shard_num)):
            isdummy, dummy, pred = self.checkForDummy(shard_i, self.y_train[self.shard_data_dict[shard_i]])
            if isdummy:
                print("dummy created")
                # dummy fit just to handle errors
                self.shard_model_dict[shard_i] = mw.modelWrapper(dummy, self.num_classes)
                self.shard_model_dict[shard_i].fit(self.X_dummy, self.y_dummy)
            else:
                self.shard_model_dict[shard_i].fit(self.X_train[self.shard_data_dict[shard_i]],
                                                   self.y_train[self.shard_data_dict[shard_i]])
        self.eclf = EnsembleVoteClassifier(clfs=list(self.shard_model_dict.values()), weights=[1] * self.num_shards,
                                           voting='hard', refit=False)
        self.eclf.fit(self.X_dummy, self.y_dummy)

    #   Creates class-balanced separations of training data and assigns to each shard
    def initialize_bookkeeping_dicts(self):
        y = self.y_train
        manager = [0] * len(Counter(y).keys())
        self.shard_data_dict = {sh_num: [] for sh_num in range(self.num_shards)}
        for it in range(len(y)):
            self.shard_data_dict[manager[y[it]]].append(it)
            self.data_to_shard_dict[it] = manager[y[it]]
            manager[y[it]] = (manager[y[it]] + 1) % self.num_shards
        self.shard_model_dict = {sh_num: mw.modelWrapper(model=self.ml_algorithm, num_classes=self.num_classes)
                                 for sh_num in range(self.num_shards)}
        self.cur_train_ids = list(range(len(y)))

    def initialize_bookkeeping_dicts2(self):
        y = self.y_train
        self.shard_data_dict = {sh_num: [] for sh_num in range(self.num_shards)}
        cur_shard = 0
        for i in range(len(y)):
            self.shard_data_dict[cur_shard].append(i)
            self.data_to_shard_dict[i] = cur_shard
            cur_shard = (cur_shard + 1) % self.num_shards
        self.shard_model_dict = {sh_num: mw.modelWrapper(model=self.ml_algorithm, num_classes=self.num_classes)
                                 for sh_num in range(self.num_shards)}
        self.cur_train_ids = list(range(len(y)))

    def getShardNum(self, idx):
        return [self.data_to_shard_dict[id_i] for id_i in idx]

    def checkForDummy(self, X, y):
        if len(y) is 0:
            return True, DummyClassifier(strategy="constant", constant=self.default_class), self.default_class
        elif len(Counter(y).keys()) is 1:
            return True, DummyClassifier(strategy="constant", constant=y[0]), y[0]
        return False, None, None


#   EnsembleShardedClassifier: extends VanillaShardedClassifier:
#       - Fits an ensemble of independent shard models using ensembleselction.EnsembleSelectionClassifier
#       - Prediction using ensemble model
#       - Unlearning followed by creating a new ensemble using retrained shard models
class EnsembleShardedClassifier(VanillaShardedClassifier):
    ensembleModel = None

    def fit(self, X, y):
        super().fit(X, y)
        self.ensembleModel = ensembleselection.EnsembleSelectionClassifier().getEnsemble(
            list(self.shard_model_dict.values()), self.X_train, self.y_train)

    def predict(self, X):
        return self.ensembleModel.predict(X)

    def unlearn(self, X_y_ids):
        super().unlearn(X_y_ids)
        self.ensembleModel = ensembleselection.EnsembleSelectionClassifier().getEnsemble(
            list(self.shard_model_dict.values()), self.X_train[self.cur_train_ids], self.y_train[self.cur_train_ids])


class TestEnsembleShardedClassifier(VanillaShardedClassifier):
    ensembleModel = None

    def fit(self, X, y):
        super().fit(X, y)
        self.ensembleModel = ensembleselection.EnsembleSelectionClassifier().getEnsemble(
            list(self.shard_model_dict.values()), self.X_train, self.y_train, ens_voting='hard',
            initial_weights=[1] * self.num_shards)

    def predict(self, X):
        return self.eclf.predict(X), self.ensembleModel.predict(X)


class TestEnsembleMultipleModels():
    ensembleModel = None

    def __init__(self, num_shards, ml_algorithm):
        if not isinstance(ml_algorithm, list):
            raise ValueError("pass a list of distinct models")
        self.num_shards = num_shards
        self.ml_algorithm = ml_algorithm
        self.X_train = None
        self.y_train = None
        self.shard_data_dict = {}
        self.shard_model_dict = {}
        self.data_to_shard_dict = {}
        self.cur_train_ids = []
        self.default_class = None
        self.num_classes = None
        self.X_dummy = None
        self.y_dummy = None
        self.eclf = None

    def fit(self, X, y):
        self.X_train = copy.deepcopy(X)
        self.y_train = copy.deepcopy(y)
        self.default_class = Counter(y).most_common(1)[0][0]
        self.num_classes = len(set(self.y_train))
        self.X_dummy = np.zeros(shape=(self.num_classes, len(X[0])))
        self.y_dummy = np.asarray(list(range(self.num_classes)))
        self.num_shards = min(self.num_shards, Counter(y).most_common()[-1][1])
        self.initialize_bookkeeping_dicts()
        for shard_num in self.shard_model_dict:
            self.shard_model_dict[shard_num].fit(self.X_train[self.shard_data_dict[shard_num]],
                                                 self.y_train[self.shard_data_dict[shard_num]])
        self.eclf = EnsembleVoteClassifier(clfs=list(self.shard_model_dict.values()),
                                           weights=[1] * self.num_shards,
                                           voting='hard', refit=False)
        self.eclf.fit(self.X_dummy, self.y_dummy)
        self.ensembleModel = ensembleselection.EnsembleSelectionClassifier().getEnsemble(
            list(self.shard_model_dict.values()), self.X_train, self.y_train, ens_voting='hard',
            initial_weights=[1] * self.num_shards)

    def initialize_bookkeeping_dicts(self):
        y = self.y_train
        manager = [0] * len(Counter(y).keys())
        self.shard_data_dict = {sh_num: [] for sh_num in range(self.num_shards)}
        for it in range(len(y)):
            self.shard_data_dict[manager[y[it]]].append(it)
            self.data_to_shard_dict[it] = manager[y[it]]
            manager[y[it]] = (manager[y[it]] + 1) % self.num_shards
        model = 0
        for it in range(self.num_shards):
            self.shard_model_dict[it] = mw.modelWrapper(model=self.ml_algorithm[model], num_classes=self.num_classes)
            model = (model + 1)%len(self.ml_algorithm)
        self.cur_train_ids = list(range(len(y)))

    def predict(self, X):
        return self.eclf.predict(X), self.ensembleModel.predict(X)


#   AutoML-Model-Reuse-Vanilla-ShardedClassifier: Extends VanillaShardedClassifier
#       - Unlearning followed by reusing (retraining) the model for each shard initially found by sklearn
#       - Suited for less unlearning requests
#       - Only one run of Bayesian optimizer for finding best models
#       - Prediction: vanilla - majority vote
class AMMRVanillaShardedClassifier(VanillaShardedClassifier):

    def __init__(self, num_shards=1, ml_algorithm=None):
        if ml_algorithm is None:
            ml_algorithm = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30, ensemble_size=1,
                                                                            include_preprocessors=['no_preprocessing'])
        elif not isinstance(ml_algorithm, autosklearn.estimators.AutoSklearnClassifier):
            raise ValueError('This classifier is only valid for '
                             'ml_algorithm=autosklearn.estimators.AutoSklearnClassifier.')
        super().__init__(num_shards, ml_algorithm)

    def fit(self, X, y):
        super().fit(X, y)
        for shard_i in range(self.num_shards):
            self.shard_model_dict[shard_i] = self.shard_model_dict[shard_i].get_models_with_weights()[0][1]


#   AutoML-Model-Reuse-Ensemble-ShardedClassifier: Extends AMMRVanillaShardedClassifier and EnsembleShardedClassifier
#       - Prediction using ensemble
#   Use following for clarity on multiple inheritance
# class vanilla:
#
#     def fit(self):
#         print("fit of vanilla")
#
#     def predict(self):
#         print("predict of vanilla")
#
#     def unlearn(self):
#         print("unlearn of vanilla")
#         self.refit()
#
#     def refit(self):
#         print("refit of vanilla")
#
# class autovanilla(vanilla):
#
#     def refit(self):
#         print("refit of autovanilla")
#
# class ensemble(vanilla):
#
#     def fit(self):
#         super().fit()
#         print("fit of ensemble")
#
#     def predict(self):
#         print("predict of ensemble")
#
#     def unlearn(self):
#         super().unlearn()
#         print("unlearn of ensemble")
#
# class autoensemble(autovanilla, ensemble):
#     pass
#
# obj = autoensemble()
# obj.fit()
# obj.predict()
# obj.unlearn()
class AMMREnsembleShardedClassifier(AMMRVanillaShardedClassifier, EnsembleShardedClassifier):

    def fit(self, X, y):
        AMMRVanillaShardedClassifier.fit(X, y)
        self.ensembleModel = ensembleselection.EnsembleSelectionClassifier().getEnsemble(
            list(self.shard_model_dict.values()), self.X_train, self.y_train)


# AMMRVSRandomVanillaShardedClassifier: Extends AMMRVanillaShardedClassifier
# Auto-ML-Model-Reuse-Validation-Set-Random-VanillaShardedClassifier
#   - Find best 'n' models on validation set using AutoML
#   - Assign best models to shards randomly
#   - Prediction and Unlearning like AMMRVanillaShardedClassifier
class AMMRVSRandomVanillaShardedClassifier(AMMRVanillaShardedClassifier):

    def __init__(self, num_shards=1):
        ml_algorithm = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=180,
                                                                        ensemble_size=num_shards,
                                                                        include_preprocessors=['no_preprocessing'])
        super().__init__(num_shards, ml_algorithm)
        self.X_val = None
        self.y_val = None

    def fit(self, X_train, y_train):
        self.X_train = copy.deepcopy(X_train)
        self.y_train = copy.deepcopy(y_train)
        self.default_class = Counter(y_train).most_common(1)[0][0]
        self.initialize_bookkeeping_dicts()
        # Initialize shards with dummy classifiers
        for shard_i in self.shard_data_dict:
            self.shard_model_dict[shard_i] = self.DummyClassifier(prediction=self.default_class)
        # Find best models based on validation set
        best_models = self.ml_algorithm.fit(self.X_train, self.y_train).get_models_with_weights()
        # Assign best models to shards
        for i in range(len(best_models)):
            self.shard_model_dict[i] = best_models[i][1]
            self.shard_model_dict[i].fit(self.X_train[self.shard_data_dict[i]],
                                         self.y_train[self.shard_data_dict[i]])
