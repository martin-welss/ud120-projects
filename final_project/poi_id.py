#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','salary', 'bonus', 'from_messages', 'to_messages', 'total_stock_value', 'exercised_stock_options'] # You will need to use more features

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X_new = SelectKBest(chi2, k=3).fit_transform(features, labels)
X_new.shape

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()

from sklearn import tree
clf = tree.DecisionTreeClassifier(min_samples_split=2)

##from sklearn.svm import SVC
##from sklearn.grid_search import GridSearchCV
##param_grid = {
##         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
##          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
##          }
##clf = GridSearchCV(SVC(kernel='rbf', class_weight='auto'), param_grid)
##clf = SVC(kernel="rbf")


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
