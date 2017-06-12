import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn_evaluation
import tabulate
import mistune

import sklearn_evaluation as skeval
from sklearn_evaluation import plot

from time import time
from datetime import datetime

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

time_now = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Starting execution @: ", time_now, " /!\ "
est = RandomForestClassifier()

training_features_scaled = np.loadtxt('out-after-features/training_features_scaled.txt', dtype=np.float64)
print "****  LOADED: training_features_scaled"
labels_array = np.loadtxt('out-after-features/labels_array.txt', dtype=int)
print "****  LOADED: labels_array"


# Utility function to report best scores


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


# use a full grid over all parameters ~2187 searches

hyperparameters4 = {
    "min_samples_leaf": [2, 6, 10],
    "min_samples_split": [10],
    'max_features': ['sqrt'],
    'n_estimators': [700]
}
# run grid search
clf = GridSearchCV(est, param_grid=hyperparameters4)
start = time()
clf.fit(training_features_scaled, labels_array)
print "=" * 30
print "=" * 30
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(clf.cv_results_['params'])))
report(clf.cv_results_)
print "=" * 30
print "=" * 30

# use a full grid over all parameters ~2187 searches
hyperparameters5 = {
    "criterion": ["gini", "entropy"],
    "min_samples_split": [10],
    'max_features': ['sqrt'],
    'n_estimators': [700]
}
# run grid search
clf = GridSearchCV(est, param_grid=hyperparameters5, cv=2)
start = time()
clf.fit(training_features_scaled, labels_array)
print "=" * 30
print "=" * 30
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(clf.cv_results_['params'])))
report(clf.cv_results_)
print "=" * 30
print "=" * 30

time_now = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Finished execution @: ", time_now, " /!\ "
