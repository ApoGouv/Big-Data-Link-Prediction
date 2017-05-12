import random


import csv

# scientific computing package | http://www.numpy.org/
import numpy as np

# Natural Language Toolkit: Natural Language Processing package | http://www.nltk.org/
import nltk

# sklearn: Machine learning package | http://scikit-learn.org/ #

# svm: Support vector machines -  are a set of supervised learning methods used for classification,
#       regression and outliers detection.
# http://scikit-learn.org/stable/modules/svm.html
from sklearn import svm

# TfidfVectorizer: Convert a collection of raw documents to a matrix of TF-IDF features.
# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
from sklearn.feature_extraction.text import TfidfVectorizer

# linear_kernel: Compute the linear kernel between X and Y.
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.linear_kernel.html
from sklearn.metrics.pairwise import linear_kernel

# preprocessing: package which provides several common utility functions and transformer classes
#               to change raw feature vectors into a representation that is more suitable for the downstream estimators.
# http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn import preprocessing

# RandomForestClassifier: A random forest classifier.
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
from sklearn.ensemble import RandomForestClassifier as RF

nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()


with open("input/testing_set.txt", "r") as f:
    reader = csv.reader(f)
    testing_set = list(reader)

testing_set = [element[0].split(" ") for element in testing_set]

###################
# random baseline #
###################

random_predictions = np.random.choice([0, 1], size=len(testing_set))
random_predictions = zip(range(len(testing_set)), random_predictions)

count_check = 0
with open("input/random_predictions.csv", "wb") as rand_pred:
    csv_out = csv.writer(rand_pred)
    print "* Write: " + "random_prediction.csv" + " - Start - ",
    for row in random_predictions:
        csv_out.writerow(row)
        count_check += 1
        if count_check % 10000 == True:
            print " . ",
    print " - Ended Successfully!!"

# read the file we just created and read the first 5 rows
with open("input/random_predictions.csv", "r") as rand_pred_file:
    csv_read = csv.reader(rand_pred_file)
    print "* READ:" + "random_prediction.csv"
    print "****", "['id', 'category']"
    for i in range(5):
        print "     ", csv_read.next()


#
# note: Kaggle requires that you add "ID" and "category" column headers
#
###############################
# beating the random baseline #
###############################
#
# the following script gets an F1 score of approximately 0.66
#
# data loading and preprocessing 
#
# the columns of the data frame below are: 
# (1) paper unique ID (integer)
# (2) publication year (integer)
# (3) paper title (string)
# (4) authors (strings separated by ,)
# (5) name of journal (optional) (string)
# (6) abstract (string) - lowercased, free of punctuation except intra-word dashes
#

#
# At this point, we read the training_set.txt and put it in a list.
# The training_set is in the form of:
#   ['src_ID  trg_ID edge']
#   ['9510123 9502114 1']
#   ['9707075 9604178 1']
#   ['9312155 9506142 0']
#   ['9911255 302165 0']
#   ['9701033 209076 0']
#
with open("input/training_set.txt", "r") as f:
    reader = csv.reader(f)
    print "****  READ:" + "training_set.txt"
    training_set = list(reader)  # Make a list from the training_set.txt

#
# Here we transform the above training_set list variable in the form of:
# ****  ['src_ID',  'trg_ID', 'edge']
#       ['9510123', '9502114', '1']
#       ['9707075', '9604178', '1']
#       ['9312155', '9506142', '0']
#       ['9911255', '302165', '0']
#       ['9701033', '209076', '0']
#
training_set = [element[0].split(" ") for element in training_set]
print "**** ", "['src_ID',  'trg_ID', 'edge']"
for i in range(5):
    print "     ", training_set[i]


with open("input/node_information.csv", "r") as f:
    reader = csv.reader(f)
    print "****  READ:" + "node_information.csv"
    node_info = list(reader)  # Make a list from the node_information.csv
    #                [0]      [1]      [2]       [3]           [4]              [5]
    print "**** ", "['ID',   'Year', 'Title', 'Authors', 'Journal name (O)', 'Abstract']"
    for i in range(5):
        print "     ", node_info[i]

IDs = [element[0] for element in node_info]  # this holds a vertical list of only the IDs

# compute TFIDF vector of each paper
corpus = [element[5] for element in node_info]  # this holds a vertical list of the Abstracts

# vectorizer initializes the TfidfVectorizer() & we can pass more parameters. see webpage in top
# stop_words="english": remove common 'english' words
# min_df=0:
# ngram_range=(1,3): generate 2 and 3 word phrases along with the single words from the corpus
# analyzer='word':
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df=0, stop_words="english")

# each row is a node in the order of node_info
# fit_transform(): Learn vocabulary and idf, return term-document matrix.
features_TFIDF = vectorizer.fit_transform(corpus)
# print type(features_TFIDF) | will print <class 'scipy.sparse.csr.csr_matrix'>
# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.sparse.csr_matrix.html

## the following shows how to construct a graph with igraph
## even though in this baseline we don't use it
## look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas

# edges = [(element[0],element[1]) for element in training_set if element[2]=="1"]

## some nodes may not be connected to any other node
## hence the need to create the nodes of the graph from node_info.csv,
## not just from the edge list

# nodes = IDs

## create empty directed graph
# g = igraph.Graph(directed=True)

## add vertices
# g.add_vertices(nodes)

## add edges
# g.add_edges(edges)




# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 5% of training set
# gives and array of IDs
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set) * 0.05)))

# produce a sample training_set
training_set_reduced = [training_set[i] for i in to_keep]
print "****  READ:" + "training_set_reduced"
#                   [0]        [1]      [2]
print "**** ", "['src_ID',  'trg_ID', 'edge']"
for i in range(5):
    print "     ", training_set_reduced[i]

# we will use three basic features:
# feature #1: number of overlapping words in title
overlap_title = []
# feature #2: temporal distance (time) between the papers
temp_diff = []
# feature #3: number of common authors
comm_auth = []
# feature #4: number of common words in journal
comm_journ = []
# feature #5: number of common abstract words
comm_abstr = []

counter = 0
# For each row in the training_set_reduced calculate the 3 features
for i in xrange(len(training_set_reduced)):

    # 1: @source: src_ID | @target: trg_ID
    source = training_set_reduced[i][0]
    target = training_set_reduced[i][1]

    # 2: @index_source and @index_target: get the corresponding index from the IDs list
    index_source = IDs.index(source)
    index_target = IDs.index(target)

    # 3: Get detail information about our source and target in the following form:
    #                   [0]        [1]      [2]       [3]             [4]             [5]
    # @source_info:  ['source',   'Year', 'Title', 'Authors', 'Journal name (O)', 'Abstract']
    # @target_info:  ['target',   'Year', 'Title', 'Authors', 'Journal name (O)', 'Abstract']
    source_info = [element for element in node_info if element[0] == source][0]
    target_info = [element for element in node_info if element[0] == target][0]

    # 4/a: Manipulate source & target title
    source_title = source_info[2].lower().split(" ")  # convert to lowercase and tokenize
    source_title = [token for token in source_title if token not in stpwds]  # remove stopwords
    source_title = [stemmer.stem(token) for token in source_title]  # perform stemming

    target_title = target_info[2].lower().split(" ")  # convert to lowercase and tokenize
    target_title = [token for token in target_title if token not in stpwds]  # remove stopwords
    target_title = [stemmer.stem(token) for token in target_title]  # perform stemming

    # 4/b: Manipulate source & target abstract
    source_abstr = source_info[5].lower().split(" ")  # convert to lowercase and tokenize
    source_abstr = [token for token in source_abstr if token not in stpwds]  # remove stopwords
    source_abstr = [stemmer.stem(token) for token in source_abstr]  # perform stemming

    target_abstr = target_info[5].lower().split(" ")  # convert to lowercase and tokenize
    target_abstr = [token for token in target_abstr if token not in stpwds]  # remove stopwords
    target_abstr = [stemmer.stem(token) for token in target_abstr]  # perform stemming


    # 4: Manipulate source & target authors
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    # 4: Manipulate source & target journal
    source_journal = source_info[4]
    target_journal = target_info[4]

    # Calculate feature #1 - number of overlapping words in title
    overlap_title.append(len(set(source_title).intersection(set(target_title))))

    # Calculate feature #2 - temporal distance (time) between the papers
    temp_diff.append(int(source_info[1]) - int(target_info[1]))

    # Calculate feature #3 - number of common authors
    comm_auth.append(len(set(source_auth).intersection(set(target_auth))))

    # Calculate feature #4 - number of common words in journal
    comm_journ.append(len(set(source_journal).intersection(set(target_journal))))

    # Calculate feature #5 - number of common abstract words
    comm_abstr.append(len(set(source_abstr).intersection(set(target_abstr))))

    counter += 1
    if counter % 10000 == True:
        print counter, " training examples processed"
    if counter % 1000 == True:
        print ".",

print " "
print " /!\ Total: ", counter, " training examples processed!"

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array([overlap_title, temp_diff, comm_auth, comm_journ, comm_abstr]).astype(np.float64).T

# scale our features
# Why apply scale?
#     If we try to apply distance based methods (such as kNN) on our features,
#     feature with the largest range will dominate the outcome results and
#     we will obtain less accurate predictions. We can overcome this trouble using feature scaling.
#
training_features_scaled = preprocessing.scale(training_features)
print "training_features" + "          " + "training_features_scaled"
for i in range(5):
    print training_features[i], "         ",  training_features_scaled[i]

# convert labels into integers then into column array
# labels: are what I call previously "edge", which says if a link between 2 papers exist (1) or not (0)
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)
print "****  READ:" + "labels_array"
print "     ", labels_array

# initialize basic SVM
# classifier = svm.LinearSVC() # SVM used in initial baseline script
# Create a random forest classifier. (By convention, clf means 'classifier')
classifier = RF(n_jobs=1, n_estimators=200, max_features=5, max_depth=10, min_samples_leaf=100)

# Train the classifier to take the training features and learn how they relate
# to the training labels_array (the edges)
classifier.fit(training_features_scaled, labels_array)

###############
# testing set #
###############
# we need to compute the features for the testing set

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
comm_journ_test = []
comm_abstr_test = []

counter = 0
# For each row in the testing_set calculate the 3 features
for i in xrange(len(testing_set)):

    # 1: @source: src_ID | @target: trg_ID
    source = testing_set[i][0]
    target = testing_set[i][1]

    # 2: @index_source and @index_target: get the corresponding index from the IDs list
    index_source = IDs.index(source)
    index_target = IDs.index(target)

    # 3: Get detail information about our source and target in the following form:
    #                   [0]        [1]      [2]       [3]             [4]             [5]
    # @source_info:  ['source',   'Year', 'Title', 'Authors', 'Journal name (O)', 'Abstract']
    # @target_info:  ['target',   'Year', 'Title', 'Authors', 'Journal name (O)', 'Abstract']
    source_info = [element for element in node_info if element[0] == source][0]
    target_info = [element for element in node_info if element[0] == target][0]

    # 4: Manipulate source & target title
    # convert to lowercase and tokenize
    source_title = source_info[2].lower().split(" ")
    # remove stopwords
    source_title = [token for token in source_title if token not in stpwds]
    # perform stemming
    source_title = [stemmer.stem(token) for token in source_title]
    # 4/a: Manipulate source & target title
    source_title = source_info[2].lower().split(" ")  # convert to lowercase and tokenize
    source_title = [token for token in source_title if token not in stpwds]  # remove stopwords
    source_title = [stemmer.stem(token) for token in source_title]  # perform stemming

    target_title = target_info[2].lower().split(" ")  # convert to lowercase and tokenize
    target_title = [token for token in target_title if token not in stpwds]  # remove stopwords
    target_title = [stemmer.stem(token) for token in target_title]  # perform stemming

    # 4/a: Manipulate source & target abstract
    source_abstr = source_info[5].lower().split(" ")  # convert to lowercase and tokenize
    source_abstr = [token for token in source_abstr if token not in stpwds]  # remove stopwords
    source_abstr = [stemmer.stem(token) for token in source_abstr]  # perform stemming

    target_abstr = target_info[5].lower().split(" ")  # convert to lowercase and tokenize
    target_abstr = [token for token in target_abstr if token not in stpwds]  # remove stopwords
    target_abstr = [stemmer.stem(token) for token in target_abstr]  # perform stemming

    # 4: Manipulate source & target authors
    source_auth = source_info[3].split(",")
    target_auth = target_info[3].split(",")

    # 5: Manipulate source & target journal
    source_journal = source_info[4]
    target_journal = target_info[4]

    # Calculate feature #1 - number of overlapping words in title
    overlap_title_test.append(len(set(source_title).intersection(set(target_title))))
    # Calculate feature #2 - temporal distance (time) between the papers
    temp_diff_test.append(int(source_info[1]) - int(target_info[1]))
    # Calculate feature #3 - number of common authors
    comm_auth_test.append(len(set(source_auth).intersection(set(target_auth))))
    # Calculate feature #4 - number of common words in journal
    comm_journ_test.append(len(set(source_journal).intersection(set(target_journal))))
    # Calculate feature #5 - number of common abstract words
    comm_abstr_test.append(len(set(source_abstr).intersection(set(target_abstr))))

    counter += 1
    if counter % 10000 == True:
        print counter, " testing examples processed"
    if counter % 1000 == True:
        print ".",

print " "
print " /!\ Total: ", counter, " testing examples processed!"

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array([overlap_title_test, temp_diff_test, comm_auth_test, comm_journ_test, comm_abstr_test]).astype(np.float64).T

# scale our features
testing_features_scaled = preprocessing.scale(testing_features)

print "testing_features" + "          " + "testing_features_scaled"
for i in range(5):
    print testing_features[i], "         ",  testing_features_scaled[i]

# issue predictions
# For our classifier: - RandomForestClassifier -
#   The predicted class of an input sample is a vote by the trees in the forest,
#   weighted by their probability estimates. That is, the predicted class is the
#   one with highest mean probability estimate across the trees.
predictions_classifier = list(classifier.predict(testing_features_scaled))

# zip: allows us to loop over multiple lists at the same time
predictions_classifier = zip(range(len(testing_set)), predictions_classifier)

# Write our prediction to a csv called: improved_predictions.csv
#   with the 1rst row, being a header with ['id', 'category'],
#   making that way 'submission ready' for Kaggle
c_ch = 0
with open("out-predictions/improved_predictions.csv", "wb") as csvfile_with_pred1:
    csv_out = csv.writer(csvfile_with_pred1)
    print "* Write: " + "improved_predictions.csv" + " - Start - ",
    csv_out.writerow(['id', 'category'])
    for row in predictions_classifier:
        csv_out.writerow(row)
        c_ch += 1
        if c_ch % 10000 == True:
            print " . ",
    print " - Ended Successfully!!"

# read the file we created above and read the first 5 rows
with open("out-predictions/improved_predictions.csv", "r") as impr_pred_file:
    csv_read = csv.reader(impr_pred_file)
    print "* READ:" + "improved_predictions.csv"
    for i in range(5):
        print "     ", csv_read.next()