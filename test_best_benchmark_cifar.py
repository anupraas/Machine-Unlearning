from sklearn import datasets, metrics, model_selection
import numpy as np
from package import shardedclassifier
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import pickle
import autosklearn.classification
from package import GenerateDataset

results_file = 'cifar10_best_benchmark'
results_ber_file = 'cifar10_best_benchmark_ber'


def preprocess_data(X, y, samplesize=None):
    le = LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    if samplesize is not None:
        print(Counter(y).most_common())
        print(len(y))
        _, X, _, y = model_selection.train_test_split(X, y, test_size=samplesize, shuffle=True, random_state=0,
                                                      stratify=y)
        print(Counter(y).most_common())
        print(len(y))
    return X, y


all_number_of_shards = [1, 5, 10, 20, 50]
MLAs = [
    "decision_tree",
    "gaussian_nb",
    "liblinear_svc"
]

X, y = GenerateDataset.CustomDataset().get_dataset('cifar10', 'cifar10')
print(len(y))
print(Counter(y).most_common())
X, y = preprocess_data(X, y, 0.1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=0,
                                                                    stratify=y)

best_models = {}
for m in MLAs:
    model = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=300,
                                                             ensemble_size=1,
                                                             ensemble_memory_limit=4096,
                                                             include_preprocessors=["no_preprocessing"],
                                                             include_estimators=[m])
    model.fit(X_train, y_train)
    best_models[m] = model.get_models_with_weights()[0][1]

unlearned_fraction = np.asarray([0, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 25, 50])
unlearn_counts = (np.rint((unlearned_fraction / 100) * len(X_train))).astype(int)
unlearn_sequence = np.asarray(range(len(X_train)))
np.random.seed(0)
np.random.shuffle(unlearn_sequence)

results = {ns: {mla: {un: None for un in unlearned_fraction} for mla in MLAs} for ns in all_number_of_shards}
results_ber = {ns: {mla: {un: None for un in unlearned_fraction} for mla in MLAs} for ns in all_number_of_shards}

for number_of_shards in all_number_of_shards:
    print("{} shards".format(number_of_shards))
    MLA_labels = []
    for mla_i in MLAs:
        print(mla_i)
        mlaname = mla_i
        MLA_labels.append(mlaname)
        sharded_mlp_results = []
        mla_i = best_models[mla_i]
        sharded_learner = shardedclassifier.VanillaShardedClassifier(number_of_shards, mla_i)
        try:
            sharded_learner.fit(X_train, y_train)
        except Exception as e:
            print("{} failed initial training".format(mlaname))
            print(e)
            continue
        predicted = sharded_learner.predict(X_test)
        initial_accuracy = metrics.accuracy_score(y_test, predicted)
        initial_accuracy_ber = metrics.balanced_accuracy_score(y_test, predicted)
        sharded_mlp_results.append(initial_accuracy)
        curmla_unlearning = [0]
        print(0)
        print(initial_accuracy)
        results[number_of_shards][mlaname][0] = initial_accuracy
        results_ber[number_of_shards][mlaname][0] = initial_accuracy_ber
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
            accuracy = metrics.accuracy_score(y_test, predicted)
            ber_accuracy = metrics.balanced_accuracy_score(y_test, predicted)
            sharded_mlp_results.append(metrics.accuracy_score(y_test, predicted))
            results[number_of_shards][mlaname][unlearned_fraction[i]] = accuracy
            results_ber[number_of_shards][mlaname][unlearned_fraction[i]] = ber_accuracy
            curmla_unlearning.append(unlearned_fraction[i])
            print(sharded_mlp_results)
            with open(results_file, 'wb') as fp:
                pickle.dump(results, fp)
            with open(results_ber_file, 'wb') as fp:
                pickle.dump(results_ber, fp)
