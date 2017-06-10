import random
import math
from datetime import datetime
import csv
import re  # regular expressions
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

import nltk

import igraph
import networkx as nx
import community
import matplotlib.pyplot as plt
import cairo

import sklearn.pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel

# preprocessing: package which provides several common utility functions and transformer classes

from sklearn import preprocessing
from sklearn.metrics import accuracy_score, log_loss

# Classifiers

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.svm.libsvm import predict_proba
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier





def warn(*args, **kwargs): pass
warnings.warn = warn

time = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Starting execution @: ", time, " /!\ "

'''''
with open("out-after-features/training_features_scaled.csv", "r") as tf:
    reader = csv.reader(tf)
    print "****  READ:" + "training_features_scaled.csv"
    training_features_scaled = list(reader)

with open("out-after-features/labels_array.csv", "r") as la:
    reader = csv.reader(la)
    print "****  READ:" + "labels_array.csv"
    labels = list(reader)
    labels_array = np.array(labels)

with open("out-after-features/testing_features_scaled.csv", "r") as tsf:
    reader = csv.reader(tsf)
    print "****  READ:" + "testing_features_scaled.csv"
    testing_features_scaled = list(reader)
'''
training_features_scaled = np.loadtxt('out-after-features/training_features_scaled.txt', dtype=np.float64)
print "****  LOADED: training_features_scaled"
labels_array = np.loadtxt('out-after-features/labels_array.txt', dtype=int)
print "****  LOADED: labels_array"
testing_features_scaled = np.loadtxt('out-after-features/testing_features_scaled.txt', dtype=np.float64)
print "****  LOADED: testing_features_scaled"

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(training_features_scaled, labels_array, test_size=0.20, random_state=42)


classifiers = [
    KNeighborsClassifier(3),
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    AdaBoostClassifier(),
    RandomForestClassifier(n_jobs=1, n_estimators=500, criterion="entropy", max_features="log2", max_depth=10),
    ExtraTreesClassifier(),
    BaggingClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LinearDiscriminantAnalysis(),
    MLPClassifier()]

# Logging for Visual Comparison
log_cols = ["Classifier", "Accuracy", "Log Loss"]
log = pd.DataFrame(columns=log_cols)

'''
print("=" * 30)
clf2 = DecisionTreeClassifier()
clf2.fit(X_train, y_train)
name = clf2.__class__.__name__
time = datetime.now().strftime('%H:%M:%S')
print "2. " + name + " starting @: " + time + "."
print '****Results****'
train_predictions = clf2.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print "Accuracy: {:.4%}".format(acc)
train_predictions = clf2.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print "Log Loss: {}".format(ll)
print("=" * 30)

print("=" * 30)
clf3 = RandomForestClassifier()
clf3.fit(X_train, y_train)
name = clf3.__class__.__name__
time = datetime.now().strftime('%H:%M:%S')
print "3. " + name + " starting @: " + time + "."
print '****Results****'
train_predictions = clf3.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print "Accuracy: {:.4%}".format(acc)
train_predictions = clf3.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print "Log Loss: {}".format(ll)
print("=" * 30)

print("=" * 30)
clf4 = AdaBoostClassifier()
clf4.fit(X_train, y_train)
name = clf4.__class__.__name__
time = datetime.now().strftime('%H:%M:%S')
print "4. " + name + " starting @: " + time + "."
print '****Results****'
train_predictions = clf4.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print "Accuracy: {:.4%}".format(acc)
train_predictions = clf4.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print "Log Loss: {}".format(ll)
print("=" * 30)

print("=" * 30)
clf5 = RandomForestClassifier(n_jobs=1, n_estimators=500, criterion="entropy", max_features="log2", max_depth=10)
clf5.fit(X_train, y_train)
name = clf5.__class__.__name__
time = datetime.now().strftime('%H:%M:%S')
print "5. " + name + " starting @: " + time + "."
print '****Results****'
train_predictions = clf5.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print "Accuracy: {:.4%}".format(acc)
train_predictions = clf5.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print "Log Loss: {}".format(ll)
print("=" * 30)

print("=" * 30)
clf6 = ExtraTreesClassifier()
clf6.fit(X_train, y_train)
name = clf6.__class__.__name__
time = datetime.now().strftime('%H:%M:%S')
print "6. " + name + " starting @: " + time + "."
print '****Results****'
train_predictions = clf6.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print "Accuracy: {:.4%}".format(acc)
train_predictions = clf6.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print "Log Loss: {}".format(ll)
print("=" * 30)

print("=" * 30)
clf7 = MLPClassifier()
clf7.fit(X_train, y_train)
name = clf7.__class__.__name__
time = datetime.now().strftime('%H:%M:%S')
print "7. " + name + " starting @: " + time + "."
print '****Results****'
train_predictions = clf7.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print "Accuracy: {:.4%}".format(acc)
train_predictions = clf7.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print "Log Loss: {}".format(ll)
print("=" * 30)

print("=" * 30)
clf8 = BaggingClassifier()
clf8.fit(X_train, y_train)
name = clf8.__class__.__name__
time = datetime.now().strftime('%H:%M:%S')
print "8. " + name + " starting @: " + time + "."
print '****Results****'
train_predictions = clf8.predict(X_test)
acc = accuracy_score(y_test, train_predictions)
print "Accuracy: {:.4%}".format(acc)
train_predictions = clf8.predict_proba(X_test)
ll = log_loss(y_test, train_predictions)
print "Log Loss: {}".format(ll)
print("=" * 30)
'''

clf_number = 0
for clf in classifiers:
    clf_number = clf_number + 1
    clf.fit(X_train, y_train)
    name = clf.__class__.__name__

    print("=" * 30)
    time = datetime.now().strftime('%H:%M:%S')
    print clf_number, ". " + name + " starting @: " + time + "."

    print '****Results****'
    train_predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, train_predictions)
    print "Accuracy: {:.4%}".format(acc)

    train_predictions = clf.predict_proba(X_test)
    ll = log_loss(y_test, train_predictions)
    print "Log Loss: {}".format(ll)

    log_entry = pd.DataFrame([[name, acc * 100, ll]], columns=log_cols)
    log = log.append(log_entry)

print("=" * 30)


sns.set_color_codes("muted")
sns.barplot(x='Accuracy', y='Classifier', data=log, color="b")

plt.xlabel('Accuracy %')
plt.title('Classifier Accuracy')
plt.savefig("out-stats-graphs/clf-accuracy.pdf")
# plt.show()

sns.set_color_codes("muted")
sns.barplot(x='Log Loss', y='Classifier', data=log, color="g")

plt.xlabel('Log Loss')
plt.title('Classifier Log Loss')
plt.savefig("out-stats-graphs/clf-loss.pdf")
# plt.show()


time = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Finished execution @: ", time, " /!\ "

