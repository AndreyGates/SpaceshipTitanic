from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import learning_curve
from sklearn.svm import SVC
from matplotlib import pyplot as plt
import numpy as np

from data_prep import prepared_data

X_train, y_train, X_test, y_test = prepared_data()
X, y = X_train[:100], y_train[:100]
estimator = AdaBoostClassifier(learning_rate=0.08, n_estimators=500)

train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator, X, y, cv=10, return_times=True)

plt.plot(train_sizes,np.mean(train_scores,axis=1))