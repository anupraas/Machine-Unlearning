from package import ensembleselection
from sklearn import datasets, metrics, model_selection, svm, neural_network as nn
from sklearn.dummy import DummyClassifier
from mlxtend.classifier import EnsembleVoteClassifier

X, y = datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, shuffle=True, random_state=1)

clf1 = svm.SVC(gamma=0.001)
clf1 = DummyClassifier()
clf2 = nn.MLPClassifier(solver='lbfgs')
clf1.fit(X_train, y_train)
print("Dummy score = {}".format(metrics.accuracy_score(y_test, clf1.predict(X_test))))
clf2.fit(X_train, y_train)
print("MLP score = {}".format(metrics.accuracy_score(y_test, clf2.predict(X_test))))
eclf, wts = ensembleselection.EnsembleSelectionClassifier().getEnsemble(models=[clf1, clf2], X=X_train, y=y_train, ret_weights=True)
print("ECLF score = {}".format(metrics.accuracy_score(y_test, eclf.predict(X_test))))
print("final wts:")
print(wts)
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2], voting='soft', weights=[0, 100], refit=False)
eclf.fit(X, y)
print("standard ECLF score = {}".format(metrics.accuracy_score(y_test, eclf.predict(X_test))))
