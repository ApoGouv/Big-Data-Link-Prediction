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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier





def warn(*args, **kwargs): pass
warnings.warn = warn

time = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Starting execution @: ", time, " /!\ "


nltk.download('punkt')  # for tokenization
nltk.download('stopwords')
stpwds = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.PorterStemmer()
stemmerRegXP = nltk.stem.RegexpStemmer(r'\([^)]*\)')  # removes text inside parenthesis & parenthesis

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
#                  [0]         [1]      [2]
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
vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words="english")

# each row is a node in the order of node_info
# fit_transform(): Learn vocabulary and idf, return term-document matrix.
features_TFIDF_Abstract = vectorizer.fit_transform(corpus)
# print type(features_TFIDF) | will print <class 'scipy.sparse.csr.csr_matrix'>
# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.sparse.csr_matrix.html

# compute TFIDF vector of each title
corpusTitle = [element[2] for element in node_info]
# each row is a node in the order of node_info
features_TFIDF_Title = vectorizer.fit_transform(corpusTitle)

# compute TFIDF vector of each author
corpusAuthor = [element[3] for element in node_info]
# each row is a node in the order of node_info
features_TFIDF_Author = vectorizer.fit_transform(corpusAuthor)

# compute TFIDF vector of each journal
corpusJournal = [element[4] for element in node_info]
# each row is a node in the order of node_info
features_TFIDF_Journal = vectorizer.fit_transform(corpusJournal)

# the following shows how to construct a graph with igraph
# even though in this baseline we don't use it
# look at http://igraph.org/python/doc/igraph.Graph-class.html for feature ideas
edges = [(element[0], element[1]) for element in training_set if element[2] == "1"]

# some nodes may not be connected to any other node
# hence the need to create the nodes of the graph from node_info.csv,
# not just from the edge list

nodes = IDs

# create empty directed graph
g = igraph.Graph(directed=True)

# add vertices
g.add_vertices(nodes)

# add edges
g.add_edges(edges)

# print g

visual_style = {}
visual_style["edge_curved"] = False
visual_style["vertex_color"] = "#AC0045"
visual_style["vertex_size"] = 0
visual_style["vertex_label_size"] = 12
visual_style["bbox"] = (1066, 800)
visual_style["margin"] = 10

# time = datetime.now().strftime('%H:%M:%S')
# print "*** Starting ploting whole graph... @:" + time
# igraph.plot(g, "complete-graph.pdf", **visual_style)
# time = datetime.now().strftime('%H:%M:%S')
# print "*** Graph ploting ENDED ... @:" + time

'''
# make the graph with the networkX library
gx = nx.Graph()  # this is undirected graph
# gx = nx.DiGraph()  # this is directed graph
gx.add_nodes_from(nodes)
gx.add_edges_from(edges)
# find communities
time = datetime.now().strftime('%H:%M:%S')
print "*** Starting community calculation... @:" + time
clusters = community.best_partition(gx)
time = datetime.now().strftime('%H:%M:%S')
print "*** Community calculation ENDED ... @:" + time
'''

# time = datetime.now().strftime('%H:%M:%S')
# print "*** Starting community ploting... @:" + time
# nx.draw_spring(gx, cmap=plt.get_cmap('jet'), node_color='#A0CBE2',edge_color='#BB0000', node_size=25, with_labels=False)
# plt.savefig("communities-graph.pdf")
# time = datetime.now().strftime('%H:%M:%S')
# print "*** Community ploting ENDED ... @:" + time

# Find Communities with iGraph (had problem)
# clusters = g.community_multilevel(return_levels=True)
# igraph.plot(clusters, "communities-graph.png", mark_groups=True, **visual_style)

# an adjacency list is a collection of unordered lists used to represent a finite graph.
# Each list describes the set of neighbors of a vertex in the graph from wikipedia.
# So we get the adjacency list of the graph and convert each item in the adjacency list to a set.
gAdjList = [set(x) for x in g.get_adjlist(mode="ALL")]

# preferential attachment | we pre-calculate here the degrees array
degrees = g.degree()

# for each training example we need to compute features
# in this baseline we will train the model on only 5% of the training set

# randomly select 80% of training set
# gives an array of IDs
to_keep = random.sample(range(len(training_set)), k=int(round(len(training_set) * 0.8)))

# produce a sample training_set
training_set_reduced = [training_set[i] for i in to_keep]
print "****  READ:" + "training_set_reduced"
#                   [0]        [1]      [2]
print "**** ", "['src_ID',  'trg_ID', 'edge']"
for i in range(5):
    print "     ", training_set_reduced[i]

# Graph for the reduced training set
# g_training = igraph.Graph(directed=True)
# IDs_training = [element[0] for element in training_set_reduced]
# IDs_training.append([element[1] for element in training_set_reduced if element[0] != element[1]])
# nodes_training = IDs_training
# edges_training = [(element[0], element[1]) for element in training_set_reduced if element[2] == "1"]
# g_training.add_vertices(nodes_training)  # add vertices
# g_training.add_edges(edges_training)  # add edges

# gAdjList_training = [set(x) for x in g_training.get_adjlist(mode="ALL")]  # adjacency list
# degrees_training = g_training.degree()  # preferential attachment degrees array

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
# feature #6a: cosine similarity
cos_sim_abstract = []
# feature #6b: cosine similarity author
cos_sim_author = []
# feature #6c:  cosine similarity title
cos_sim_title = []
# feature #6d:  cosine similarity journal
cos_sim_journal = []
# feature #7: common neighbours
com_neigh = []
# feature #8: preferential attachment
pref_attach = []
# feature #9: Jaccard similarity coefficient
jac_sim = []


def jaccard_coefficent(s, d, g):
    s_neighbors = set(g.neighbors(s))
    d_neighbors = set(g.neighbors(d))
    intersection = len(s_neighbors.intersection(d_neighbors))
    union = len(s_neighbors.union(d_neighbors))
    if union == 0:
        return 0.0
    else:
        return float(intersection / float(union))


# feature #9: Adamic Adar similarity
adam_adar = []


def adamic_adar(s, d, g):
    s_neighbors = set(g.neighbors(s))
    d_neighbors = set(g.neighbors(d))
    aa = 0.0
    for i in s_neighbors.intersection(d_neighbors):
        if math.log(len(g.neighbors(i))) == 0:
            aa += 0.0
        else:
            aa += float(1 / math.log(len(g.neighbors(i))))
    return aa


# feature #?: shortest distance
# see paper: Link prediction using supervised learning @p.4 | 3.Topological Featutes: Shortest Distance
# shortest_path = []
# feature #10: Calculates the Google PageRank values of a graph.
# g.pagerank(g, vertices=None, directed=True, damping=0.85, weights=None, arpack_options=None, implementation='prpack', niter=1000, eps=0.001)
page_rank = []
page_rank = g.pagerank()
page_rank_list_target = []
page_rank_list_source = []

# feature #11: check if in the same cluster
# calculate dendrogram
# dendrogram  = g.community_edge_betweenness(directed=True)
# convert it into a flat clustering
# clusters = dendrogram.as_clustering()
# get the membership vector
# membership = clusters.membership
# partition = communities.best_partition(g)
'''
same_cluster = []


def is_same_cluster(s, d, clusters):
    # clusters is alist of VertexClustering (VC) objetcs
    if clusters.get(s) == clusters.get(d):
        return 1
    else:
        return 0
'''

counter = 0
time = datetime.now().strftime('%H:%M:%S')
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
    source_title = [stemmerRegXP.stem(token) for token in source_title]  # perform stemming for parenthesis
    source_title = [stemmer.stem(token) for token in source_title]  # perform stemming

    target_title = target_info[2].lower().split(" ")  # convert to lowercase and tokenize
    target_title = [token for token in target_title if token not in stpwds]  # remove stopwords
    target_title = [stemmerRegXP.stem(token) for token in target_title]  # perform stemming for parenthesis
    target_title = [stemmer.stem(token) for token in target_title]  # perform stemming

    # 4/b: Manipulate source & target abstract
    source_abstr = source_info[5].lower().split(" ")  # convert to lowercase and tokenize
    source_abstr = [token for token in source_abstr if token not in stpwds]  # remove stopwords
    source_abstr = [stemmer.stem(token) for token in source_abstr]  # perform stemming

    target_abstr = target_info[5].lower().split(" ")  # convert to lowercase and tokenize
    target_abstr = [token for token in target_abstr if token not in stpwds]  # remove stopwords
    target_abstr = [stemmer.stem(token) for token in target_abstr]  # perform stemming

    # 4: Manipulate source & target authors
    source_auth = source_info[3]
    source_auth = re.sub(r'\([^)]*\)', '', source_auth)  # remove parenthesis and content inside them
    # if source_info[3].count("(") > 0:
    #   print " "
    #   print " *********************  "
    #   print "Author list after regEx = ", source_auth
    source_auth = source_auth.split(",")
    # if source_info[3].count("(") > 0:
    #   print " "
    #   print "Author list after split = ", source_auth
    source_auth = [stemmerRegXP.stem(token) for token in source_auth]  # perform stemming for parenthesis
    # if source_info[3].count("(") > 0:
    #    print " "
    #    print "Author w/e parenthesis from stemmer= ", source_auth
    source_auth[:] = [val for val in source_auth if not val == " " or val == ""]  # remove empty entries in our list
    # for sa, val in enumerate(source_auth):
    #    if val == " " or val == "":
    #        del source_auth[sa]
    # iterate through our author list end call strip() to remove starting and trailing spaces
    for sa, val in enumerate(source_auth):
        source_auth[sa] = source_auth[sa].strip()
    # if source_info[3].count("(") > 0:
    #    print " "
    #    print "Author list sripped = ", source_auth
    #    print " *********************  "

    target_auth = target_info[3]
    target_auth = re.sub(r'\([^)]*\)', '', target_auth)
    target_auth = target_auth.split(",")
    target_auth = [stemmerRegXP.stem(token) for token in target_auth]
    target_auth[:] = [val for val in target_auth if not val == " " or val == ""]
    for ta, val in enumerate(target_auth):
        target_auth[ta] = target_auth[ta].strip()

    # 5: Manipulate source & target journal
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
    # Calculate feature #6a - abstract cosine similarity
    cos_sim_abstract.append(
        cosine_similarity(features_TFIDF_Abstract[index_source], features_TFIDF_Abstract[index_target]))
    # Calculate feature #6b - title cosine similarity
    cos_sim_title.append(cosine_similarity(features_TFIDF_Title[index_source], features_TFIDF_Title[index_target]))
    # Calculate feature #6c - author cosine similarity
    cos_sim_author.append(cosine_similarity(features_TFIDF_Author[index_source], features_TFIDF_Author[index_target]))
    # Calculate feature #6d - journal cosine similarity
    cos_sim_journal.append(
        cosine_similarity(features_TFIDF_Journal[index_source], features_TFIDF_Journal[index_target]))
    # Calculate feature #7: common neighbours
    com_neigh.append(len(gAdjList[index_source].intersection(gAdjList[index_target])))
    # Calculate feature #8: preferential attachment
    pref_attach.append(int(degrees[index_source] * degrees[index_target]))
    # Calculate feature #9: Jaccard similarity
    jac_sim.append(jaccard_coefficent(index_source, index_target, g))
    # Calculate feature #9: Adamic Adar similarity
    adam_adar.append(adamic_adar(index_source, index_target, g))
    # Calculate feature #? - shortest path
    # cg.get_shortest_paths(v=index_source, to=index_target, weights=None, mode=1, output= )
    # shortest_path.append(
    #   len(g.shortest_paths_dijkstra(source=index_source, target=index_target, weights=None, mode=1)))
    # Calculate feature #10,11: pagerank source, target
    page_rank_list_target.append(page_rank[index_target])
    page_rank_list_source.append(page_rank[index_source])

    # same_cluster.append(is_same_cluster(index_source, index_target, clusters))

    counter += 1
    if counter % 10000 == True:
        time = datetime.now().strftime('%H:%M:%S')
        print counter, " training examples processed, @: ", time
    if counter % 1000 == True:
        print ".",

time = datetime.now().strftime('%H:%M:%S')
print " "
print " /!\ Total: ", counter, " training examples processed! @: ", time

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
training_features = np.array(
    [overlap_title, temp_diff, comm_auth, comm_journ, comm_abstr, cos_sim_abstract, cos_sim_author,
     cos_sim_journal, cos_sim_title, com_neigh, pref_attach, jac_sim, adam_adar, page_rank_list_source,
     page_rank_list_target]).astype(np.float64).T

np.savetxt('out-after-features/training_features.txt', training_features)
print "****  SAVED: training_features"

# scale our features
# Why apply scale?
#     If we try to apply distance based methods (such as kNN) on our features,
#     feature with the largest range will dominate the outcome results and
#     we will obtain less accurate predictions. We can overcome this trouble using feature scaling.
#
training_features_scaled = preprocessing.scale(training_features, copy=False)


np.savetxt('out-after-features/training_features_scaled.txt', training_features_scaled)
print "****  SAVED: training_features_scaled"

'''
count_check = 0
with open("out-after-features/training_features_scaled.csv", "wb") as tr_feat_scale:
    csv_out = csv.writer(tr_feat_scale)
    print "* Write: " + "training_features_scaled.csv" + " - Start - ",
    for row in training_features_scaled:
        csv_out.writerow(row)
        count_check += 1
        if count_check % 10000 == True:
            print " . ",
    print " - Ended Successfully!!"
'''

# print "training_features" + "          " + "training_features_scaled"
# for i in range(5):
#     print training_features[i], "         ", training_features_scaled[i]

# convert labels into integers then into column array
# labels: are what I call previously "edge", which says if a link between 2 papers exist (1) or not (0)
labels = [int(element[2]) for element in training_set_reduced]
labels = list(labels)
labels_array = np.array(labels)
print "****  READ:" + "labels_array"
print "     ", labels_array


np.savetxt('out-after-features/labels_array.txt', labels_array)
print "****  SAVED: labels_array"
###############
# testing set #
###############
# we need to compute the features for the testing set

overlap_title_test = []
temp_diff_test = []
comm_auth_test = []
comm_journ_test = []
comm_abstr_test = []
cos_sim_abstract_test = []
cos_sim_author_test = []
cos_sim_journal_test = []
cos_sim_title_test = []
com_neigh_test = []
pref_attach_test = []
jac_sim_test = []
adam_adar_test = []
# shortest_path_test = []
page_rank_list_source_test = []
page_rank_list_target_test = []
# same_cluster_test = []

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
    source_title = [stemmerRegXP.stem(token) for token in source_title]  # perform stemming
    source_title = [stemmer.stem(token) for token in source_title]  # perform stemming

    target_title = target_info[2].lower().split(" ")  # convert to lowercase and tokenize
    target_title = [token for token in target_title if token not in stpwds]  # remove stopwords
    target_title = [stemmerRegXP.stem(token) for token in target_title]  # perform stemming
    target_title = [stemmer.stem(token) for token in target_title]  # perform stemming

    # 4/a: Manipulate source & target abstract
    source_abstr = source_info[5].lower().split(" ")  # convert to lowercase and tokenize
    source_abstr = [token for token in source_abstr if token not in stpwds]  # remove stopwords
    source_abstr = [stemmer.stem(token) for token in source_abstr]  # perform stemming

    target_abstr = target_info[5].lower().split(" ")  # convert to lowercase and tokenize
    target_abstr = [token for token in target_abstr if token not in stpwds]  # remove stopwords
    target_abstr = [stemmer.stem(token) for token in target_abstr]  # perform stemming

    # 4: Manipulate source & target authors

    source_auth = target_info[3]
    source_auth = re.sub(r'\([^)]*\)', '', source_auth)
    source_auth = source_auth.split(",")
    source_auth = [stemmerRegXP.stem(token) for token in source_auth]
    source_auth[:] = [val for val in source_auth if not val == " " or val == ""]
    for ta, val in enumerate(source_auth):
        source_auth[ta] = source_auth[ta].strip()

    target_auth = target_info[3]
    target_auth = re.sub(r'\([^)]*\)', '', target_auth)
    target_auth = target_auth.split(",")
    target_auth = [stemmerRegXP.stem(token) for token in target_auth]
    target_auth[:] = [val for val in target_auth if not val == " " or val == ""]
    for ta, val in enumerate(target_auth):
        target_auth[ta] = target_auth[ta].strip()

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
    # Calculate feature #6a - abstract cosine similarity
    cos_sim_abstract_test.append(
        cosine_similarity(features_TFIDF_Abstract[index_source], features_TFIDF_Abstract[index_target]))
    # Calculate feature #6b - title cosine similarity
    cos_sim_title_test.append(cosine_similarity(features_TFIDF_Title[index_source], features_TFIDF_Title[index_target]))
    # Calculate feature #6c - author cosine similarity
    cos_sim_author_test.append(
        cosine_similarity(features_TFIDF_Author[index_source], features_TFIDF_Author[index_target]))
    # Calculate feature #6d - journal cosine similarity
    cos_sim_journal_test.append(
        cosine_similarity(features_TFIDF_Journal[index_source], features_TFIDF_Journal[index_target]))

    # Calculate feature #7: common neighbours
    com_neigh_test.append(len(gAdjList[index_source].intersection(gAdjList[index_target])))
    # Calculate feature #8: preferential attachment
    pref_attach_test.append(int(degrees[index_source] * degrees[index_target]))
    # Calculate feature #9: Jaccard similarity
    jac_sim_test.append(jaccard_coefficent(index_source, index_target, g))
    # Calculate feature #10: Adamic Adar similarity
    adam_adar_test.append(adamic_adar(index_source, index_target, g))
    # Calculate feature #? - shortest path
    # shortest_path_test.append(
    #   len(g.shortest_paths_dijkstra(source=index_source, target=index_target, weights=None, mode=1)))
    # Calculate feature #10,11: pagerank source, target
    page_rank_list_target_test.append(page_rank[index_target])
    page_rank_list_source_test.append(page_rank[index_source])

    # same_cluster_test.append(is_same_cluster(index_source, index_target, clusters))

    counter += 1
    if counter % 10000 == True:
        time = datetime.now().strftime('%H:%M:%S')
        print counter, " testing examples processed, @: ", time
    if counter % 1000 == True:
        print ".",

time = datetime.now().strftime('%H:%M:%S')
print " "
print " /!\ Total: ", counter, " testing examples processed! @: ", time

# convert list of lists into array
# documents as rows, unique words as columns (i.e., example as rows, features as columns)
testing_features = np.array(
    [overlap_title_test, temp_diff_test, comm_auth_test, comm_journ_test, comm_abstr_test, cos_sim_abstract_test,
     cos_sim_author_test, cos_sim_journal_test, cos_sim_title_test, com_neigh_test, pref_attach_test,
     jac_sim_test, adam_adar_test, page_rank_list_source_test, page_rank_list_target_test]).astype(
    np.float64).T

# scale our features
testing_features_scaled = preprocessing.scale(testing_features, copy=False)

np.savetxt('out-after-features/testing_features_scaled.txt', testing_features_scaled)
print "****  SAVED: testing_features_scaled"

'''
count_check = 0
with open("out-after-features/testing_features_scaled.csv", "wb") as ts_feat_scale:
    csv_out = csv.writer(ts_feat_scale)
    print "* Write: " + "testing_features_scaled.csv" + " - Start - ",
    for row in testing_features_scaled:
        csv_out.writerow(row)
        count_check += 1
        if count_check % 10000 == True:
            print " . ",
    print " - Ended Successfully!!"
'''

time = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Finished execution @: ", time, " /!\ "

