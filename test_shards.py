from sklearn import datasets, metrics, model_selection, svm, neural_network as nn
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
from collections import Counter
import autosklearn.classification


class DummyClassifier:
    prediction = 0

    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return [self.prediction] * len(X)


class ShardedClassifier:
    num_shards = 1
    ml_algorithm = None
    X_train = None
    y_train = None
    shard_data_dict = {}
    shard_model_dict = {}
    data_to_shard_dict = {}
    init_shard_size = 0
    ensemble_model = None
    default_class = None

    def __init__(self, num_shards, ml_algorithm):
        self.num_shards = num_shards
        self.ml_algorithm = ml_algorithm

    def fit(self, X, y, default_class=0):
        self.X_train = copy.deepcopy(X)
        self.y_train = copy.deepcopy(y)
        self.init_shard_size = len(y) // self.num_shards
        self.default_class = default_class
        self.initialize_dicts(self.X_train, self.y_train)
        for shard_num in self.shard_data_dict:
            self.shard_model_dict[shard_num] = self.fit_shard(self.X_train[self.shard_data_dict[shard_num]],
                                                              self.y_train[self.shard_data_dict[shard_num]])

    def fit_shard(self, X, y):
        if len(X) is 0:
            return DummyClassifier(prediction=self.default_class)
        elif len(Counter(y).keys()) is 1:
            return DummyClassifier(prediction=y[0])
        else:
            return self.ml_algorithm.fit(X, y)

    def predict(self, X):
        predictions = []
        for m in self.shard_model_dict:
            predictions.append(self.shard_model_dict[m].predict(X))
        predictions = np.asarray(predictions)
        ret_predictions = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(len(X))]
        return ret_predictions

    def initialize_dicts(self, X, y):
        manager = [0] * len(Counter(y).keys())
        self.shard_data_dict = {sh_num: [] for sh_num in range(self.num_shards)}
        for it in range(len(y)):
            self.shard_data_dict[manager[y[it]]].append(it)
            self.data_to_shard_dict[it] = manager[y[it]]
            manager[y[it]] = (manager[y[it]] + 1) % self.num_shards

    def getShardNum(self, id):
        return [self.data_to_shard_dict[id_i] for id_i in id]

    def unlearn(self, X_y_ids):
        shard_num = self.getShardNum(X_y_ids)
        for i in range(len(X_y_ids)):
            self.shard_data_dict[shard_num[i]].remove(X_y_ids[i])
        for shard_i in list(set(shard_num)):
            self.shard_model_dict[shard_i] = self.fit_shard(self.X_train[self.shard_data_dict[shard_i]],
                                                            self.y_train[self.shard_data_dict[shard_i]])


number_of_shards = 3
# MLAs = [svm.SVC(gamma=0.001), nn.MLPClassifier(solver='lbfgs')]
# MLA_labels = ['SVM', 'MLP']
# MLAs = [svm.SVC(gamma=0.001), nn.MLPClassifier(solver='lbfgs'), autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30,
#                                                                                                                  ensemble_size=1,
#                                                                                                                  initial_configurations_via_metalearning=0)]
# MLA_labels = ['SVM', 'MLP', 'Auto-Sklearn']
# MLAs = [autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30, ensemble_size=1, initial_configurations_via_metalearning=0, include_preprocessors=['no_preprocessing'])]
MLAs = [autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30, ensemble_size=1, include_preprocessors=['no_preprocessing'])]
MLA_labels = ['Auto-Sklearn']
X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)
unlearned_fraction = np.asarray([0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
unlearn_counts = (np.rint((unlearned_fraction / 100) * len(X_train))).astype(int)
unlearn_sequence = np.asarray(range(len(X_train)))
np.random.shuffle(unlearn_sequence)

for mla_i in range(len(MLAs)):
    print(MLAs[mla_i])
    sharded_mlp_results = []
    # sharded_learner = ShardedClassifier(10, nn.MLPClassifier(solver='lbfgs'))
    # sharded_learner = ShardedClassifier(10, svm.SVC(gamma=0.001))
    # sharded_learner = ShardedClassifier(10, autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30, ensemble_size=1))
    sharded_learner = ShardedClassifier(number_of_shards, MLAs[mla_i])
    sharded_learner.fit(X_train, y_train)
    predicted = sharded_learner.predict(X_test)
    initial_accuracy = metrics.accuracy_score(y_test, predicted)
    sharded_mlp_results.append(initial_accuracy)
    print(0)
    print(initial_accuracy)
    for i in range(1, len(unlearn_counts)):
        frac = unlearn_counts[i]
        prev_frac = unlearn_counts[i - 1]
        inds = unlearn_sequence[prev_frac:frac]
        print(frac)
        sharded_learner.unlearn(inds)
        predicted = sharded_learner.predict(X_test)
        sharded_mlp_results.append(metrics.accuracy_score(y_test, predicted))
        print(sharded_mlp_results)
    plt.plot(unlearned_fraction, sharded_mlp_results)
plt.legend(MLA_labels)
plt.xlabel('% points unlearned')
plt.ylabel('Accuracy')
plt.title('{} Shards'.format(number_of_shards))
plt.show()
