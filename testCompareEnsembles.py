from sklearn import metrics, model_selection
from package import shardedclassifier as sc
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from package import GenerateDataset
import pickle

results_file = 'covtype_auto_part_2'
results_ber_file = 'covtype_auto_part_2_ber'


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


all_shards = [5, 10, 20]
MLAs = [
    # AdaBoostClassifier(),
    # BernoulliNB(),
    DecisionTreeClassifier(),
    # ExtraTreesClassifier(),
    GaussianNB(),
    # GradientBoostingClassifier(),
    # KNeighborsClassifier(1),
    # LinearDiscriminantAnalysis(),
    # MultinomialNB(),
    # PassiveAggressiveClassifier(),
    # QuadraticDiscriminantAnalysis(),
    LinearSVC(),
    MLPClassifier(alpha=1, max_iter=500),
    # SVC(gamma=2, C=1)
    # MultinomialNB(),
    # RandomForestClassifier(5),
    # SGDClassifier()
]


X, y = GenerateDataset.CustomDataset().get_dataset('mnist', 'mnist')
X, y = preprocess_data(X, y, 0.05)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

results = {ns: {mla.__class__.__name__: None for mla in MLAs} for ns in all_shards}
results_ber = {ns: {mla.__class__.__name__: None for mla in MLAs} for ns in all_shards}

for num_shards in all_shards:
    for mla in MLAs:
        print(mla)
        sharded_results = []
        mla_name = mla.__class__.__name__
        sharded_learner = sc.TestEnsembleShardedClassifier(num_shards, mla)
        sharded_learner.fit(X_train, y_train)
        predicted, predicted_ens = sharded_learner.predict(X_test)
        results[num_shards][mla_name] = (metrics.accuracy_score(y_test, predicted),
                                         metrics.accuracy_score(y_test, predicted))
        results_ber[num_shards][mla_name] = (metrics.balanced_accuracy_score(y_test, predicted_ens),
                                             metrics.balanced_accuracy_score(y_test, predicted_ens))
        print(results[num_shards][mla_name])
        with open(results_file, 'wb') as fp:
            pickle.dump(results, fp)
        with open(results_ber_file, 'wb') as fp:
            pickle.dump(results_ber, fp)
