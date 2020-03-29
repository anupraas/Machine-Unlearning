import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import numpy as np

#X, y = sklearn.datasets.load_digits(n_class=2, return_X_y=True)
X, y = sklearn.datasets.load_digits(return_X_y=True)
print("Dataset loaded")
print("X: {}".format(np.shape(X)))
print("y: {}".format(np.shape(y)))
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

automl = autosklearn.classification.AutoSklearnClassifier(time_left_for_this_task=30)
automl.fit(X_train, y_train)
print("Model trained")
y_hat = automl.predict(X_test)
print("Predictions done")
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))
print(automl.show_models())
