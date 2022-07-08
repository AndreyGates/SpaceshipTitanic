from sklearn.model_selection import train_test_split
import xgboost as xgb
import numpy as np

from data_prep import data_extract, prepared_data

from hpsklearn import HyperoptEstimator, xgboost_classification
from hpsklearn import any_classifier
from hpsklearn import any_preprocessing
from hyperopt import tpe

X_train, y_train, X_test, y_test = prepared_data()
X, y = X_train[:], y_train[:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


'''USING HYPEROPTIMIZER TO FIND THE BEST MODEL'''
if __name__ == '__main__':
    model = HyperoptEstimator(classifier=any_classifier('clf'), 
                              preprocessing=any_preprocessing('pre'), 
                              max_evals=100, 
                              trial_timeout=60)
    model.fit(X_train, y_train)

    # summarize performance
    acc = model.score(X_test, y_test)
    print("Accuracy: %.3f" % acc)
    # summarize the best model
    print(model.best_model())