import autosklearn.classification
from sklearn import datasets, metrics, model_selection, svm, neural_network as nn
import matplotlib.pyplot as plt
import random

X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True)
unlearned_fraction = [1, 2, 5, 10, 20, 25, 40, 50, 60, 75, 80, 90, 95, 99]
sksvm_results = []
skmlp_results = []
autosk_results = []

prev_frac = 0
print(prev_frac)
print(len(X_train))
for frac in unlearned_fraction:
    inds = set(random.sample(list(range(len(X_train))), int((frac-prev_frac)/100 * len(X_train))))
    prev_frac = frac
    X_train = [n for i, n in enumerate(X_train) if i not in inds]
    y_train = [n for i, n in enumerate(y_train) if i not in inds]
    print(frac)
    print(len(X_train))
    # classifier = svm.SVC()
    # classifier.fit(X_train, y_train)
    # predicted = classifier.predict(X_test)
    # sksvm_results.append(metrics.accuracy_score(y_test, predicted))
    classifier = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30, ensemble_size=1)
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    autosk_results.append(metrics.accuracy_score(y_test, predicted))
    classifier = nn.MLPClassifier()
    classifier.fit(X_train, y_train)
    predicted = classifier.predict(X_test)
    skmlp_results.append(metrics.accuracy_score(y_test, predicted))
    print(sksvm_results)
    print(skmlp_results)
    print(autosk_results)

# plt.plot(unlearned_fraction, sksvm_results)
plt.plot(unlearned_fraction, skmlp_results)
plt.plot(unlearned_fraction, autosk_results)
# plt.legend(['sklearn-svm', 'sklearn-mlp', 'auto_sklearn'])
plt.legend(['sklearn-mlp', 'auto_sklearn'])
plt.xlabel('% points unlearned')
plt.ylabel('Accuracy')
plt.show()
