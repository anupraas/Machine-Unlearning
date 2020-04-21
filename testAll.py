from sklearn import datasets, metrics, model_selection
import matplotlib.pyplot as plt
import numpy as np
from package import shardedclassifier, autoshardedclassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import random
from collections import Counter
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.preprocessing import LabelEncoder
# from package import GenerateDataset


def preprocess_data(X, y, samplesize=None):
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    if samplesize is not None:
        print(Counter(y).most_common())
        print(len(y))
        _, X, _, y = model_selection.train_test_split(X, y, test_size=samplesize, shuffle=True, random_state=0, stratify=y)
        print(Counter(y).most_common())
        print(len(y))
    return X, y


all_number_of_shards = [1, 5, 10, 20, 50, 100, 200]
MLAs = [
    autoshardedclassifier.AutoShardedClassifier(),
    # AdaBoostClassifier(),
    # BernoulliNB(),
    DecisionTreeClassifier(),
    # ExtraTreesClassifier(),
    GaussianNB(),
    # GradientBoostingClassifier(),
    KNeighborsClassifier(1),
    # LinearDiscriminantAnalysis(),
    # MultinomialNB(),
    # PassiveAggressiveClassifier(),
    # QuadraticDiscriminantAnalysis(),
    # LinearSVC(),
    MLPClassifier(),
    SVC(),
    # MultinomialNB(),
    # PassiveAggressiveClassifier(),
    # QuadraticDiscriminantAnalysis(),
    # RandomForestClassifier(5),
    # SGDClassifier()
]

# X, y = datasets.fetch_kddcup99(shuffle=True, random_state=0, return_X_y=True)
X, y = datasets.fetch_covtype(return_X_y=True, shuffle=True, random_state=0)
# X, y = GenerateDataset.CustomDataset().get_dataset('cifar10', 'cifar10')
# X, y = datasets.load_digits(return_X_y=True)
X, y = preprocess_data(X, y, 0.1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0, stratify=y)
unlearned_fraction = np.asarray([0, 1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99])
# unlearned_fraction = np.asarray([0, 1, 2, 5])
unlearn_counts = (np.rint((unlearned_fraction / 100) * len(X_train))).astype(int)
unlearn_sequence = np.asarray(range(len(X_train)))
np.random.seed(0)
np.random.shuffle(unlearn_sequence)

initial_accuracies = {mla_i.__class__.__name__: [] for mla_i in MLAs}

for number_of_shards in all_number_of_shards:
    MLA_labels = []
    for mla_i in MLAs:
        print(mla_i)
        mlaname = mla_i.__class__.__name__
        MLA_labels.append(mlaname)
        sharded_mlp_results = []
        if mlaname is 'AutoShardedClassifier':
            sharded_learner = autoshardedclassifier.AutoShardedClassifier(number_of_shards)
        else:
            sharded_learner = shardedclassifier.VanillaShardedClassifier(number_of_shards, mla_i)
        try:
            sharded_learner.fit(X_train, y_train)
        except Exception as e:
            print("{} failed initial training".format(mlaname))
            print(e)
            continue
        predicted = sharded_learner.predict(X_test)
        initial_accuracy = metrics.accuracy_score(y_test, predicted)
        sharded_mlp_results.append(initial_accuracy)
        curmla_unlearning = [0]
        print(0)
        print(initial_accuracy)
        initial_accuracies[mlaname].append(initial_accuracy)
        for i in range(1, len(unlearn_counts)):
            frac = unlearn_counts[i]
            prev_frac = unlearn_counts[i - 1]
            inds = unlearn_sequence[prev_frac:frac]
            print('unlearning {}% : {} points'.format(unlearned_fraction[i], frac))
            try:
                sharded_learner.unlearn(inds)
            except Exception as e:
                print("{} failed while unlearning {}% points".format(mlaname, unlearned_fraction[i]))
                print(e)
                break
            predicted = sharded_learner.predict(X_test)
            sharded_mlp_results.append(metrics.accuracy_score(y_test, predicted))
            curmla_unlearning.append(unlearned_fraction[i])
            print(sharded_mlp_results)
        if mlaname is 'AutoShardedClassifier':
            plt.plot(curmla_unlearning, sharded_mlp_results, linewidth=3, color='black')
        else:
            plt.plot(curmla_unlearning, sharded_mlp_results)
    plt.xlabel('% points unlearned')
    plt.ylabel('Accuracy')
    plt.title('{} Shards'.format(number_of_shards))
    plt.savefig('{}-shards.png'.format(number_of_shards))
    plt.legend(MLA_labels)
    plt.savefig('{}-shards-with-legend.png'.format(number_of_shards))
    plt.clf()
valid_names = []
for mla in initial_accuracies:
    if len(initial_accuracies[mla]) > 0:
        if mla is 'AutoShardedClassifier':
            plt.plot(all_number_of_shards[:len(initial_accuracies[mla])], initial_accuracies[mla], linewidth=3, color='black')
        else:
            plt.plot(all_number_of_shards[:len(initial_accuracies[mla])], initial_accuracies[mla])
        valid_names.append(mla)
plt.xlabel('Total shards')
plt.ylabel('Initial Accuracy')
plt.savefig('Initial-Accuracies.png')
plt.legend(valid_names)
plt.savefig('Initial-Accuracies-with-legend.png')
plt.clf()
