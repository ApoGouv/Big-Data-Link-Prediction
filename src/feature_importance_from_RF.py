import csv
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import sklearn_evaluation as skeval
import tabulate
import mistune

from matplotlib import axes as ax
from sklearn_evaluation import plot

from sklearn.ensemble import RandomForestClassifier

time = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Starting execution @: ", time, " /!\ "

with open("input/testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]
training_features_scaled = np.loadtxt('out-after-features/training_features_scaled.txt', dtype=np.float64)
print "****  LOADED: training_features_scaled"
labels_array = np.loadtxt('out-after-features/labels_array.txt', dtype=int)
print "****  LOADED: labels_array"
testing_features_scaled = np.loadtxt('out-after-features/testing_features_scaled.txt', dtype=np.float64)
print "****  LOADED: testing_features_scaled"

classifier = RandomForestClassifier(n_jobs=1, n_estimators=700, criterion="gini", min_samples_split=10,
                                    min_samples_leaf=2, max_features="sqrt", max_depth=10)

# Train the classifier to take the training features and learn how they relate
# to the training labels_array (the edges)
classifier.fit(training_features_scaled, labels_array)

feature_importance = list(zip(training_features_scaled, classifier.feature_importances_))
predictions_classifier = list(classifier.predict(testing_features_scaled))

# zip: allows us to loop over multiple lists at the same time

predictions_classifier = zip(range(len(testing_set)), predictions_classifier)
'''
feature_names = ['overlap_title', 'temp_diff', 'comm_auth', 'comm_journ', 'comm_abstr', 'cos_sim_abstract',
                 'cos_sim_author',
                 'cos_sim_journal', 'cos_sim_title', 'com_neigh', 'pref_attach', 'jac_sim', 'adam_adar',
                 'page_rank_list_source',
                 'page_rank_list_target']
'''
feature_names = ['Overlapping words in titles', 'Temporal distance between papers', 'Number of common authors',
                 'Overlapping words in journal', 'Overlapping words in abstract', 'Cosine similarity of abstract',
                 'Cosine similarity of author', 'Cosine similarity of journal', 'Cosine similarity of title',
                 'Common neighbours', 'Preferential attachment', 'Jaccard similarity', 'Adamic Adar similarity',
                 'Pagerank from source', 'Pagerank from target']

'''  Styles for the plot
[u'seaborn-darkgrid',
 u'seaborn-notebook',
 u'classic',
 u'seaborn-ticks',
 u'grayscale',
 u'bmh',
 u'seaborn-talk',
 u'dark_background',
 u'ggplot',
 u'fivethirtyeight',
 u'seaborn-colorblind',
 u'seaborn-deep',
 u'seaborn-whitegrid',
 u'seaborn-bright',
 u'seaborn-poster',
 u'seaborn-muted',
 u'seaborn-paper',
 u'seaborn-white',
 u'seaborn-pastel',
 u'seaborn-dark',
 u'seaborn-dark-palette']
'''
my_dpi = 96
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = (11.69, 8.27)
plot.feature_importances(classifier, feature_names=feature_names)
# plt.xlabel('Feature Names')
plt.ylabel('Feature Importance Score (%)')
# plt.title('Features Importance')
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
plt.xticks(rotation=90)
f = plt.gcf()
f.subplots_adjust(bottom=0.4)
plt.savefig("out-stats-graphs/RF_Feature_Importance1.pdf")

plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (1164 / my_dpi, 1024 / my_dpi)
plot.feature_importances(classifier, feature_names=feature_names)
plt.ylabel('Feature Importance Score (%)')
plt.xlabel('Feature Names')
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
plt.xticks(rotation=90)
f = plt.gcf()
f.subplots_adjust(bottom=0.4)
plt.savefig("out-stats-graphs/RF_Feature_Importance2.pdf", bbox_inches='tight', dpi=100)

plt.style.use('seaborn-deep')
plt.rcParams["figure.figsize"] = (11.69, 8.27)
plot.feature_importances(classifier, feature_names=feature_names)
plt.ylabel('Feature Importance Score (%)')
plt.xlabel('Feature Names')
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
plt.xticks(rotation=90)
f = plt.gcf()
f.subplots_adjust(bottom=0.4)
plt.savefig("out-stats-graphs/RF_Feature_Importance3.pdf")

plt.style.use('seaborn-dark-palette')
plot.feature_importances(classifier, feature_names=feature_names)
plt.ylabel('Feature Importance Score (%)')
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
plt.xticks(rotation=90)
plt.rc('figure', figsize=(11.69, 8.27))
f = plt.gcf()
f.subplots_adjust(bottom=0.4)
plt.savefig("out-stats-graphs/RF_Feature_Importance4.pdf")

plt.style.use('seaborn-muted')
plot.feature_importances(classifier, feature_names=feature_names)
plt.ylabel('Feature Importance Score (%)')
plt.gca().xaxis.set_minor_formatter(ticker.NullFormatter())
plt.xticks(rotation=90)
f = plt.gcf()
f.subplots_adjust(bottom=0.4)
plt.savefig("out-stats-graphs/RF_Feature_Importance5.pdf", bbox_inches='tight', dpi=100)

time = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Finished execution @: ", time, " /!\ "
