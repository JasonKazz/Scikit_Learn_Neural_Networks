from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV

# data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y)

# algorithm
model = MLPClassifier(solver='sgd', max_iter=10000)
param_grid = {'hidden_layer_sizes': [(2,),(2,3),(5,5)], 'learning_rate_init': [0.01,0.001]}
search = GridSearchCV(model, param_grid, verbose=4)

# training
search.fit(X_train,y_train)

# testing
y_pred = search.predict(X_test)

# analysis
acc = accuracy_score(y_test, y_pred)
print(acc)

print(search.best_params_)
print(search.best_score_)
print(search.best_estimator_)
print(search.best_index_)
dir(search)
search.cv_results_
