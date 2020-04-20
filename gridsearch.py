from sklearn.neural_network import MLPClassifier
import random
from sklearn.model_selection import GridSearchCV
from sklearn import datasets, metrics, model_selection, svm
import numpy as np


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


# mlp = MLPClassifier(max_iter=100)
# parameter_space = {
#     'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
#     'activation': ['tanh', 'relu'],
#     'solver': ['sgd', 'adam', 'lbfgs'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant','adaptive'],
# }

mlp = svm.SVC()

parameter_space = {
    'gamma': ['scale', 'auto', 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

X, y = datasets.fetch_covtype(return_X_y=True, shuffle=True, random_state=1)
y = y - 1
X, y = preprocess_covtype(X, y, 2500)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)
clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

# Best paramete set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

y_true, y_pred = y_test , clf.predict(X_test)

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))
