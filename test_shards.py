from sklearn import datasets, metrics, model_selection, svm, neural_network as nn
import matplotlib.pyplot as plt
import random
import copy
import numpy as np
from collections import Counter


class ShardedClassifier:
    num_shards = 1
    ml_algorithm = None
    X_train = None
    y_train = None
    shard_data_dict = {}
    shard_model_dict = {}
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
        for it in range(0, len(y), self.init_shard_size):
            shard_num = self.getShardNum(it)
            self.shard_data_dict[shard_num] = list(range(it, min(it + self.init_shard_size, len(y))))
            self.shard_model_dict[shard_num] = self.fit_shard(self.X_train[self.shard_data_dict[shard_num]],
                                                              self.y_train[self.shard_data_dict[shard_num]])

    def fit_shard(self, X, y):
        if len(X) > 0:
            return self.ml_algorithm.fit(X, y)
        else:
            return None

    def predict(self, X):
        predictions = []
        for m in self.shard_model_dict:
            if self.shard_model_dict[m] is not None:
                predictions.append(self.shard_model_dict[m].predict(X))
            else:
                predictions.append(np.asarray([self.default_class]*len(X)))
        predictions = np.asarray(predictions)
        ret_predictions = [Counter(predictions[:, i]).most_common(1)[0][0] for i in range(len(X))]
        return ret_predictions

    def getShardNum(self, id):
        return np.asarray(id) // self.init_shard_size

    def unlearn(self, X_y_ids):
        shard_num = self.getShardNum(X_y_ids)
        for i in range(len(X_y_ids)):
            self.shard_data_dict[shard_num[i]].remove(X_y_ids[i])
        for shard_i in list(set(shard_num)):
            self.shard_model_dict[shard_i] = self.fit_shard(self.X_train[self.shard_data_dict[shard_i]],
                                                            self.y_train[self.shard_data_dict[shard_i]])


X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)
unlearned_fraction = np.asarray([0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
sharded_mlp_results = []
sharded_learner = ShardedClassifier(10, nn.MLPClassifier(solver='lbfgs'))
# sharded_learner = ShardedClassifier(3, svm.SVC(gamma=0.001))
sharded_learner.fit(X_train, y_train)
predicted = sharded_learner.predict(X_test)
initial_accuracy = metrics.accuracy_score(y_test, predicted)
sharded_mlp_results.append(initial_accuracy)
print(0)
print(initial_accuracy)
unlearn_indices = (np.rint((unlearned_fraction/100) * len(X_train))).astype(int)
unlearn_sequence = np.asarray(range(len(X_train)))
np.random.shuffle(unlearn_sequence)
for i in range(1, len(unlearn_indices)):
    frac = unlearn_indices[i]
    prev_frac = unlearn_indices[i-1]
    inds = unlearn_sequence[prev_frac:frac]
    print(frac)
    sharded_learner.unlearn(inds)
    predicted = sharded_learner.predict(X_test)
    sharded_mlp_results.append(metrics.accuracy_score(y_test, predicted))
    print(sharded_mlp_results)
plt.plot(unlearned_fraction, sharded_mlp_results)
plt.legend(['sharded-sklearn-mlp'])
plt.xlabel('% points unlearned')
plt.ylabel('Accuracy')
plt.show()
