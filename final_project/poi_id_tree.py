#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'deferred_income', 'exercised_stock_options', 'from_messages']


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)
data_dict.pop('THE TRAVEL AGENCY IN THE PARK',0)
### Task 3: Create new feature(s)

for person in data_dict:
    # bonus by salary
    bonus=data_dict[person]['bonus']
    salary=data_dict[person]['salary']
    if bonus!="NaN" and salary!="NaN":
        data_dict[person]['bonus_by_salary']=float(bonus)/salary
    else:
        data_dict[person]['bonus_by_salary']="NaN"

    # fraction of emails from poi
    from_all=data_dict[person]['from_messages']
    from_poi=data_dict[person]['from_poi_to_this_person']
    if from_all!="NaN" and from_poi!="NaN":
        data_dict[person]['email_fraction_from_poi']=float(from_poi)/from_all
    else:
        data_dict[person]['email_fraction_from_poi']="NaN"
    


features_list.append('bonus_by_salary')
#features_list.append('email_fraction_from_poi')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
param_grid = {
    'min_samples_split': [2, 3, 4],
    'max_depth': [3,4,5],
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random']
}
clf = GridSearchCV(DecisionTreeClassifier(max_features='auto'), param_grid)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
