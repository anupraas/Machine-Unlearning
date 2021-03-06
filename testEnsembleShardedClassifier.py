from sklearn import datasets, metrics, model_selection, svm, neural_network as nn
import matplotlib.pyplot as plt
import numpy as np
from package import shardedclassifier as sc
import random
from sklearn.ensemble import RandomForestClassifier
from collections import Counter


def preprocess_covtype(X, y, num_per_class):
    total_classes = len(set(y))
    random.seed(0)
    ret_x = []
    ret_y = []
    for ind in range(total_classes):
        indices = [i for i, x in enumerate(y) if x == ind]
        rand_sample = random.sample(indices, num_per_class)
        ret_y.extend(y[rand_sample])
        ret_x.extend(X[rand_sample])
    return np.asarray(ret_x), np.asarray(ret_y)


max_number_of_shards = 10
MLAs = [nn.MLPClassifier(solver='lbfgs'), RandomForestClassifier(n_estimators=5)]
MLA_labels = ['MLPEns', 'RFEns']
X, y = datasets.fetch_covtype(return_X_y=True, shuffle=True, random_state=1)
y = y - 1
print(Counter(y).most_common())
X, y = preprocess_covtype(X, y, 2500)
print(Counter(y).most_common())
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
unlearned_fraction = np.asarray([0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
unlearn_counts = (np.rint((unlearned_fraction / 100) * len(X_train))).astype(int)
unlearn_sequence = np.asarray(range(len(X_train)))
np.random.seed(0)
np.random.shuffle(unlearn_sequence)
shardedMLAs = [sc.EnsembleShardedClassifier(max_number_of_shards, MLAs[i]) for i in range(len(MLAs))]
for smla in shardedMLAs:
    sharded_results = []
    sharded_learner = smla
    sharded_learner.fit(X_train, y_train)
    predicted = sharded_learner.predict(X_test)
    initial_accuracy = metrics.accuracy_score(y_test, predicted)
    sharded_results.append(initial_accuracy)
    print(0)
    print(initial_accuracy)
    for i in range(1, len(unlearn_counts)):
        frac = unlearn_counts[i]
        prev_frac = unlearn_counts[i - 1]
        inds = unlearn_sequence[prev_frac:frac]
        print(frac)
        sharded_learner.unlearn(inds)
        predicted = sharded_learner.predict(X_test)
        sharded_results.append(metrics.accuracy_score(y_test, predicted))
        print(sharded_results)
    plt.plot(unlearned_fraction, sharded_results)
plt.legend(MLA_labels)
plt.xlabel('% points unlearned')
plt.ylabel('Accuracy')
plt.title('{} Shards'.format(max_number_of_shards))
plt.show()
