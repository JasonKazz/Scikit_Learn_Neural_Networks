"""
Binary Classification using neural networks in Scikit-learn

Classification means dividing the data into different groups, or "classes" as opposed to regression.
Both are branches of supervised learning.
Supervised learning uses both the features, X, and the targets/labels, y.
Unsupervised learning uses just the features, X.

"""

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

#data

X,y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_repeated=0, n_clusters_per_class=1, class_sep=3)
X_train, X_test, y_train, y_test = train_test_split(X,y)

'''
X_train = np.array([[1,0], [1,2], [0,2], [5,4], [5,6]]) # training features
y_train = [0, 0, 0, 1, 1] # class labels also called targets

X_test = np.array([[0,1], [4,5], [2,2], [4,4]]) # unseen new features
y_test = [0, 1 ,0, 1] # answers we are trying to predict (we aren't supposed to know them)
'''

#algorithm
model = MLPClassifier(hidden_layer_sizes=(2,), solver='sgd', learning_rate_init=0.1, max_iter=1000) # this works well with make_classification that we used above (100 samples, 2 features)

#training
model.fit(X_train, y_train)

#testing/prediction
y_pred = model.predict(X_test)

#analysis/plotting
acc = accuracy_score(y_test, y_pred)
print(acc)

plt.scatter(X_train[:,0], X_train[:,1], c=y_train, cmap='cool', label='Training data', s=10)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred, cmap='cool', marker='x', label='Testing data', s=60)
plt.colorbar()
plt.legend()
