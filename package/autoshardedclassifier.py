import autosklearn.classification
from collections import Counter
import numpy as np
import copy
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.dummy import DummyClassifier
from package import modelwrapper as mw


class AutoShardedClassifier:

    def __init__(self, num_shards=np.inf):
        self.ml_algorithm = None
        self.num_shards = num_shards
        self.X_train = None
        self.y_train = None
        self.shard_data_dict = {}
        self.shard_model_dict = {}
        self.shard_model_weight_dict = {}
        self.data_to_shard_dict = {}
        self.cur_train_ids = []
        self.default_class = None
        self.ensemble = None
        self.num_classes = None

    def fit(self, X, y):
        self.X_train = copy.deepcopy(X)
        self.y_train = copy.deepcopy(y)
        self.num_classes = len(set(self.y_train))
        self.default_class = Counter(y).most_common(1)[0][0]
        print(self.num_shards)
        print(Counter(y).most_common()[-1][1])
        self.num_shards = min(self.num_shards, Counter(y).most_common()[-1][1])
        print(self.num_shards)
        self.ml_algorithm = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=180,
                                                                             ensemble_size=self.num_shards,
                                                                             include_preprocessors=['no_preprocessing'])
        best_models_ensemble = self.ml_algorithm.fit(self.X_train, self.y_train).get_models_with_weights()
        self.num_shards = min(self.num_shards, len(best_models_ensemble))
        print(self.num_shards)
        self.create_training_subsets_for_shards()
        best_models = [best_models_ensemble[i][1] for i in range(self.num_shards)]
        best_model_wts = np.asarray([best_models_ensemble[i][0] for i in range(self.num_shards)])
        best_model_wts = best_model_wts / np.sum(best_model_wts)
        # Fit shards
        for shard_num in range(self.num_shards):
            self.shard_model_dict[shard_num] = mw.modelWrapper(best_models[shard_num], self.num_classes)
            self.shard_model_dict[shard_num].fit(self.X_train[self.shard_data_dict[shard_num]],
                                                 self.y_train[self.shard_data_dict[shard_num]])
            self.shard_model_weight_dict[shard_num] = best_model_wts[shard_num]
        # Create ensemble
        self.ensemble = EnsembleVoteClassifier(clfs=list(self.shard_model_dict.values()),
                                               voting='soft',
                                               weights=list(self.shard_model_weight_dict.values()),
                                               refit=False)
        self.ensemble.fit(self.X_train[self.cur_train_ids], self.y_train[self.cur_train_ids])

    def predict(self, X):
        return self.ensemble.predict(X)

    def unlearn(self, X_y_ids):
        shard_num = self.getShardNum(X_y_ids)
        for i in range(len(X_y_ids)):
            self.shard_data_dict[shard_num[i]].remove(X_y_ids[i])
            self.cur_train_ids.remove(X_y_ids[i])
        self.default_class = Counter(self.y_train[self.cur_train_ids]).most_common(1)[0][0]
        # Refitting shards after unlearning - vanilla implementation: call fit() for every shard's model
        for shard_i in list(set(shard_num)):
            isDummy, dummy_model, pred = self.checkForDummy(self.X_train[self.shard_data_dict[shard_i]],
                                                            self.y_train[self.shard_data_dict[shard_i]])
            if isDummy:
                print("dummy created")
                # dummy fit just to handle errors
                dummy_model.fit(self.X_train[self.cur_train_ids], self.y_train[self.cur_train_ids])
                self.shard_model_dict[shard_i] = mw.modelWrapper(dummy_model, self.num_classes, [pred])
            else:
                self.shard_model_dict[shard_i].fit(self.X_train[self.shard_data_dict[shard_i]],
                                                   self.y_train[self.shard_data_dict[shard_i]])
        # Create ensemble
        self.ensemble = EnsembleVoteClassifier(clfs=list(self.shard_model_dict.values()),
                                               voting='soft',
                                               weights=list(self.shard_model_weight_dict.values()),
                                               refit=False)
        self.ensemble.fit(self.X_train[self.cur_train_ids], self.y_train[self.cur_train_ids])

    def create_training_subsets_for_shards(self):
        y = self.y_train
        manager = [0] * len(Counter(y).keys())
        self.shard_data_dict = {sh_num: [] for sh_num in range(self.num_shards)}
        for it in range(len(y)):
            self.shard_data_dict[manager[y[it]]].append(it)
            self.data_to_shard_dict[it] = manager[y[it]]
            manager[y[it]] = (manager[y[it]] + 1) % self.num_shards
        self.cur_train_ids = list(range(len(y)))

    def getShardNum(self, idx):
        return [self.data_to_shard_dict[id_i] for id_i in idx]

    def checkForDummy(self, X, y):
        if len(y) is 0:
            return True, DummyClassifier(strategy="constant", constant=self.default_class), self.default_class
        elif len(Counter(y).keys()) is 1:
            return True, DummyClassifier(strategy="constant", constant=y[0]), y[0]
        return False, None, None
