#!/usr/bin/python


"""
    starter code for the evaluation mini-project
    start by copying your trained/tested POI identifier from
    that you built in the validation mini-project

    the second step toward building your POI identifier!

    start by loading/formatting the data

"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 


from sklearn import cross_validation
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(
    features, labels, test_size=0.3, random_state=42)


from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)


pred=clf.predict(features_test)
print "number of pois predicted in the testset: ",sum(pred)
print "total number of people in the testset:", len(pred)

from sklearn.metrics import accuracy_score
acc = accuracy_score(pred, labels_test)
print "accuracy: ",acc

zero_pred = [0.0]*29
acc = accuracy_score(zero_pred, labels_test)
print "zero accuracy: ",acc

from sklearn.metrics import precision_score, recall_score
precision = precision_score(labels_test, pred)
print "precision: ", precision
recall = recall_score(labels_test, pred)
print "recall: ", recall

y_pred = [0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1] 
y_true = [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0]
precision = precision_score(y_true, y_pred)
print "precision: ", precision
recall = recall_score(y_true, y_pred)
print "recall: ", recall

#F1 score is good indicator for precision and recall

