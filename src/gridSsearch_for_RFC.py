import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn_evaluation
import tabulate
import mistune

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


def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, plot_name):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2), len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx, :], '-o', label=name_param_2 + ': ' + str(val))

    ax.set_title("Grid Search Scores", fontsize=20, fontweight='bold')
    ax.set_xlabel(name_param_1, fontsize=16)
    ax.set_ylabel('CV Average Score', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.savefig("out-stats-graphs/" + plot_name + ".pdf")

# use a full grid over all parameters ~2187 searches
hyperparameters1 = {
    'n_estimators': [200, 500, 700]
}
# run grid search
clf = GridSearchCV(est, param_grid=hyperparameters1, cv=3)
start = time()
clf.fit(training_features_scaled, labels_array)
print "=" * 30
print "=" * 30
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(clf.cv_results_['params'])))
report(clf.cv_results_)



# Calling Plot Method
print "=" * 30
plot_grid_search(clf.cv_results_, clf.cv_results_.n_estimators, clf.cv_results_.max_features, 'N Estimators', 'Max Features', 'RF-Estimators-mFeat')



'''
# use a full grid over all parameters ~2187 searches
hyperparameters2 = {
    "min_samples_split": [2, 6, 10],
    'n_estimators': [200, 500, 700]
}
# run grid search
clf = GridSearchCV(est, param_grid=hyperparameters2, cv=3)
start = time()
clf.fit(training_features_scaled, labels_array)
print "=" * 30
print "=" * 30
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(clf.cv_results_['params'])))
report(clf.cv_results_)

plot.grid_search(clf.cv_results_, change='n_estimators',
                subset={'n_estimators': [200, 500, 700]},
                kind='bar')
plot.savefig("out-stats-graphs/RFC-minSampleSplit-nEstimators.pdf")
# summarize the results of the grid search
print "=" * 30
print "1."
print "Best Score: " + clf.best_score_
print "Best Estimator Alpha:" + clf.best_estimator_.alpha
print "Best parameters: " + clf.best_params_


# use a full grid over all parameters ~2187 searches
hyperparameters3 = {
    "max_features": ["auto", "sqrt", "log2"],
    'n_estimators': [200, 500, 700]
}
# run grid search
clf = GridSearchCV(est, param_grid=hyperparameters3, cv=3)
start = time()
clf.fit(training_features_scaled, labels_array)
print "=" * 30
print "=" * 30
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(clf.cv_results_['params'])))
report(clf.cv_results_)
# summarize the results of the grid search
print "=" * 30
print "1."
print "Best Score: " + clf.best_score_
print "Best Estimator Alpha:" + clf.best_estimator_.alpha
print "Best parameters: " + clf.best_params_


# use a full grid over all parameters ~2187 searches
hyperparameters4 = {
    "min_samples_leaf": [2, 6, 10],
    'n_estimators': [200, 500, 700]
}
# run grid search
clf = GridSearchCV(est, param_grid=hyperparameters4, cv=3)
start = time()
clf.fit(training_features_scaled, labels_array)
print "=" * 30
print "=" * 30
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(clf.cv_results_['params'])))
report(clf.cv_results_)
# summarize the results of the grid search
print "=" * 30
print "1."
print "Best Score: " + clf.best_score_
print "Best Estimator Alpha:" + clf.best_estimator_.alpha
print "Best parameters: " + clf.best_params_



# use a full grid over all parameters ~2187 searches
hyperparameters5 = {
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"],
    'n_estimators': [200, 500, 700]
}
# run grid search
clf = GridSearchCV(est, param_grid=hyperparameters5, cv=3)
start = time()
clf.fit(training_features_scaled, labels_array)
print "=" * 30
print "=" * 30
print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
      % (time() - start, len(clf.cv_results_['params'])))
report(clf.cv_results_)
# summarize the results of the grid search
print "=" * 30
print "1."
print "Best Score: " + clf.best_score_
print "Best Estimator Alpha:" + clf.best_estimator_.alpha
print "Best parameters: " + clf.best_params_
'''

time_now = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Finished execution @: ", time_now, " /!\ "
