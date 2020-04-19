from sklearn import datasets, metrics, model_selection, svm, neural_network as nn
import matplotlib.pyplot as plt
import numpy as np
from package import autoshardedclassifier as asc
from package import shardedclassifier as sc
import autosklearn.classification
from sklearn.ensemble import RandomForestClassifier


max_number_of_shards = 140
MLAs = [svm.SVC(gamma=0.001),
        nn.MLPClassifier(solver='lbfgs'),
        RandomForestClassifier(n_estimators=5)]
MLA_labels = ['AutoEnsAS', 'SVM', 'MLP', 'RF']
X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
unlearned_fraction = np.asarray([0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
unlearn_counts = (np.rint((unlearned_fraction / 100) * len(X_train))).astype(int)
unlearn_sequence = np.asarray(range(len(X_train)))
np.random.seed(0)
np.random.shuffle(unlearn_sequence)

ascsharded_results = []
sharded_learner = asc.AutoShardedClassifier(max_number_of_shards, ensemble_strategy=1)
sharded_learner.fit(X_train, y_train)
predicted = sharded_learner.predict(X_test)
initial_accuracy = metrics.accuracy_score(y_test, predicted)
ascsharded_results.append(initial_accuracy)
print(0)
print(initial_accuracy)
for i in range(1, len(unlearn_counts)):
    frac = unlearn_counts[i]
    prev_frac = unlearn_counts[i - 1]
    inds = unlearn_sequence[prev_frac:frac]
    print(frac)
    sharded_learner.unlearn(inds)
    predicted = sharded_learner.predict(X_test)
    ascsharded_results.append(metrics.accuracy_score(y_test, predicted))
    print(ascsharded_results)
plt.plot(unlearned_fraction, ascsharded_results)
shardedMLAs = [sc.VanillaShardedClassifier(max_number_of_shards, MLAs[i]) for i in range(len(MLAs))]
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
