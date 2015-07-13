#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")
from itertools import combinations
from sklearn.naive_bayes import GaussianNB

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

class FeatureSet:
    def __init__(self, f1, precision, recall, accuracy, features):
        self.__f1=f1
        self.__precision=precision
        self.__recall=recall
        self.__accuracy=accuracy
        self.__features=features[1:]

    def show(self):
        print "%.3f\t%.3f\t%.3f\t%.3f\t%s" % (self.__f1, self.__precision, self.__recall, self.__accuracy, self.__features)


    def __lt__(self, other):
        return self.__f1<other.__f1

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary', 'bonus', 'total_stock_value', 'exercised_stock_options']
all_features = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances',
                'bonus', 'restricted_stock_deferred', 'deferred_income',
                'total_stock_value', 'expenses', 'from_poi_to_this_person',
                'exercised_stock_options', 'from_messages', 'other', 'from_this_person_to_poi',
                'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock',
                'director_fees']


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
    


all_features.append('bonus_by_salary')
all_features.append('email_fraction_from_poi')

### Store to my_dataset for easy export below.
my_dataset = data_dict

result=[]
for features_list in combinations(all_features, 4):
    # 'poi' must always be the first feature
    features_list=['poi']+list(features_list)

    ### Extract features and labels from dataset for local testing
    data = featureFormat(my_dataset, features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)


    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    clf = GaussianNB()


    ### Task 5: Tune your classifier to achieve better than .3 precision and recall 
    ### using our testing script.
    ### Because of the small size of the dataset, the script uses stratified
    ### shuffle split cross validation. For more info: 
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    try:
        (accuracy, precision, recall, f1, f2)=test_classifier(clf, my_dataset, features_list)
        if precision>0.3 and recall>0.3:
            result.append(FeatureSet(f1, precision, recall, accuracy, features_list))
            print "result count: ", len(result)
    except:
        pass

for fs in sorted(result, reverse=True):
    fs.show()

