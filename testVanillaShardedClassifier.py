from sklearn import datasets, metrics, model_selection, svm, neural_network as nn
import matplotlib.pyplot as plt
import numpy as np
from package import shardedclassifier


number_of_shards = 3
MLAs = [svm.SVC(gamma=0.001), nn.MLPClassifier(solver='lbfgs')]
MLA_labels = ['SVM', 'MLP']
X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)
unlearned_fraction = np.asarray([0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
unlearn_counts = (np.rint((unlearned_fraction / 100) * len(X_train))).astype(int)
unlearn_sequence = np.asarray(range(len(X_train)))
np.random.shuffle(unlearn_sequence)

for mla_i in range(len(MLAs)):
    print(MLAs[mla_i])
    sharded_mlp_results = []
    sharded_learner = shardedclassifier.VanillaShardedClassifier(number_of_shards, MLAs[mla_i])
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
