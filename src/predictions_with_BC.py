import csv
import numpy as np
from datetime import datetime

from sklearn.ensemble import BaggingClassifier

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

classifier = BaggingClassifier()

# Train the classifier to take the training features and learn how they relate
# to the training labels_array (the edges)
classifier.fit(training_features_scaled, labels_array)


predictions_classifier = list(classifier.predict(testing_features_scaled))

# zip: allows us to loop over multiple lists at the same time
predictions_classifier = zip(range(len(testing_set)), predictions_classifier)

# Write our prediction to a csv called: improved_predictions.csv
#   with the 1rst row, being a header with ['id', 'category'],
#   making that way 'submission ready' for Kaggle
c_ch = 0
with open("out-predictions/BC_predictions.csv", "wb") as csvfile_with_pred1:
    csv_out = csv.writer(csvfile_with_pred1)
    print "* Write: " + "BC_predictions.csv" + " - Start - ",
    csv_out.writerow(['id', 'category'])
    for row in predictions_classifier:
        csv_out.writerow(row)
        c_ch += 1
        if c_ch % 10000 == True:
            print " . ",
    print " - Ended Successfully!!"

# read the file we created above and read the first 5 rows
with open("out-predictions/BC_predictions.csv", "r") as impr_pred_file:
    csv_read = csv.reader(impr_pred_file)
    print "* READ:" + "BC_predictions.csv"
    for i in range(5):
        print "     ", csv_read.next()

time = datetime.now().strftime('%H:%M:%S')
print " /!\ Script Finished execution @: ", time, " /!\ "
