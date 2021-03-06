import autosklearn.classification
from collections import Counter
import numpy as np
import copy
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.dummy import DummyClassifier
from package import modelwrapper as mw, ensembleselection as es
from sklearn.model_selection import StratifiedKFold

class AutoShardedClassifier:

    def __init__(self, num_shards=np.inf, ensemble_strategy=1):
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
        if ensemble_strategy not in [1, 2, 3, 4, 5]:
            ensemble_strategy = 1
        self.ensemble_strategy = ensemble_strategy
        self.X_dummy = None
        self.y_dummy = None
        if ensemble_strategy is 4:
            self.all_ensembles = [None] * 3

    def fit(self, X, y):
        self.X_train = copy.deepcopy(X)
        self.y_train = copy.deepcopy(y)
        self.num_classes = len(set(self.y_train))
        self.X_dummy = np.zeros(shape=(self.num_classes, len(X[0])))
        self.y_dummy = np.asarray(list(range(self.num_classes)))
        self.default_class = Counter(y).most_common(1)[0][0]
        # print(self.num_shards)
        # print(Counter(y).most_common()[-1][1])
        self.num_shards = min(self.num_shards, Counter(y).most_common()[-1][1])
        # print(self.num_shards)
        self.create_training_subsets_for_shards()
        self.ml_algorithm = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=900,
                                                                             ensemble_size=self.num_shards,
                                                                             # ensemble_nbest=max(2*self.num_shards, 50),
                                                                             ensemble_memory_limit=4096,
                                                                             include_preprocessors=["no_preprocessing"],
                                                                             exclude_estimators=[                                                                             
                                                                             # "random_forest", 
                                                                             # "adaboost", 
                                                                             # "extra_trees", 
                                                                             # "gradient_boosting",
                                                                             "k_nearest_neighbors"]
                                                                             )
        best_models_ensemble = self.ml_algorithm.fit(self.X_train, self.y_train).get_models_with_weights()
        shards_model_assignment_sequence = list(range(self.num_shards))
        shards_model_assignment_sequence_idx = 0
        for i in range(len(best_models_ensemble)):
            num_cur_model_shards = int(self.num_shards * best_models_ensemble[i][0])
            for j in range(num_cur_model_shards):
                shard_num = shards_model_assignment_sequence[shards_model_assignment_sequence_idx]
                self.shard_model_dict[shard_num] = mw.modelWrapper(best_models_ensemble[i][1], self.num_classes)
                self.shard_model_dict[shard_num].fit(self.X_train[self.shard_data_dict[shard_num]],
                                                     self.y_train[self.shard_data_dict[shard_num]])
                self.shard_model_weight_dict[shard_num] = 1
                shards_model_assignment_sequence_idx += 1

        # Create ensemble
        if self.ensemble_strategy is 4:
            self.create_all_ensembles()
        else:
            self.create_ensemble()

    def predict(self, X):
        if self.ensemble_strategy is 4:
            pred = []
            for i in range(3):
                pred.append(self.all_ensembles[i].predict(X))
            return pred
        return self.ensemble.predict(X)

    def unlearn(self, X_y_ids):
        shard_num = self.getShardNum(X_y_ids)
        for i in range(len(X_y_ids)):
            self.shard_data_dict[shard_num[i]].remove(X_y_ids[i])
            self.cur_train_ids.remove(X_y_ids[i])
        self.default_class = Counter(self.y_train[self.cur_train_ids]).most_common(1)[0][0]
        # Refitting shards after unlearning - vanilla implementation: call fit() for every shard's model
        for shard_i in list(set(shard_num)):
            isDummy, dummy_model, pred = self.checkForDummy(self.y_train[self.shard_data_dict[shard_i]])
            if isDummy:
                print("dummy created")
                # dummy fit just to handle errors
                self.shard_model_dict[shard_i] = mw.modelWrapper(dummy_model, self.num_classes)
                self.shard_model_dict[shard_i].fit(self.X_dummy, self.y_dummy)
            else:
                self.shard_model_dict[shard_i].fit(self.X_train[self.shard_data_dict[shard_i]],
                                                   self.y_train[self.shard_data_dict[shard_i]])
        # Create ensemble
        if self.ensemble_strategy is 4:
            self.create_all_ensembles()
        elif self.ensemble_strategy is not 5:
            self.create_ensemble()

    def create_training_subsets_for_shards(self):
        self.cur_train_ids = list(range(len(self.y_train)))
        if self.num_shards is 1:
            self.shard_data_dict[0] = copy.deepcopy(self.cur_train_ids)
            self.data_to_shard_dict = {it: 0 for it in range(len(self.y_train))}
        else:
            skf = StratifiedKFold(n_splits=self.num_shards, shuffle=True, random_state=0)
            shard_num = 0
            for train_idx, test_idx in skf.split(self.X_train, self.y_train):
                self.shard_data_dict[shard_num] = copy.deepcopy(test_idx.tolist())
                for i in test_idx:
                    self.data_to_shard_dict[i] = shard_num
                shard_num += 1

    def create_ensemble(self):
        if self.ensemble_strategy is 1:
            self.ensemble = EnsembleVoteClassifier(clfs=list(self.shard_model_dict.values()),
                                                   voting='soft',
                                                   weights=list(self.shard_model_weight_dict.values()),
                                                   refit=False)
            self.ensemble.fit(self.X_dummy, self.y_dummy)
        elif self.ensemble_strategy is 2:
            self.ensemble = es.EnsembleSelectionClassifier(maxIter=np.inf).getEnsemble(
                models=list(self.shard_model_dict.values()),
                X=self.X_train[self.cur_train_ids],
                y=self.y_train[self.cur_train_ids])
            self.ensemble.fit(self.X_dummy, self.y_dummy)
        else:
            self.ensemble, new_weights = es.EnsembleSelectionClassifier(maxIter=np.inf).getEnsemble(
                models=list(self.shard_model_dict.values()),
                X=self.X_train[self.cur_train_ids],
                y=self.y_train[self.cur_train_ids],
                initial_weights=list(self.shard_model_weight_dict.values()),
                ret_weights=True)
            self.ensemble.fit(self.X_dummy, self.y_dummy)
            # now update weights
            for i in range(self.num_shards):
                self.shard_model_weight_dict[i] = new_weights[i]

    def create_all_ensembles(self):
        self.all_ensembles[0] = EnsembleVoteClassifier(clfs=list(self.shard_model_dict.values()),
                                                       voting='soft',
                                                       weights=[1] * self.num_shards,
                                                       refit=False)
        self.all_ensembles[0].fit(self.X_dummy, self.y_dummy)

        self.all_ensembles[1] = es.EnsembleSelectionClassifier(maxIter=np.inf).getEnsemble(
            models=list(self.shard_model_dict.values()),
            X=self.X_train[self.cur_train_ids],
            y=self.y_train[self.cur_train_ids])
        self.all_ensembles[1].fit(self.X_dummy, self.y_dummy)

        self.all_ensembles[2], new_weights = es.EnsembleSelectionClassifier(maxIter=np.inf).getEnsemble(
            models=list(self.shard_model_dict.values()),
            X=self.X_train[self.cur_train_ids],
            y=self.y_train[self.cur_train_ids],
            initial_weights=list(self.shard_model_weight_dict.values()),
            ret_weights=True)
        self.all_ensembles[2].fit(self.X_dummy, self.y_dummy)
        # now update weights
        for i in range(self.num_shards):
            self.shard_model_weight_dict[i] = new_weights[i]

    def getShardNum(self, idx):
        return [self.data_to_shard_dict[id_i] for id_i in idx]

    def checkForDummy(self, y):
        if len(y) is 0:
            return True, DummyClassifier(strategy="constant", constant=self.default_class), self.default_class
        elif len(Counter(y).keys()) is 1:
            return True, DummyClassifier(strategy="constant", constant=y[0]), y[0]
        return False, None, None
