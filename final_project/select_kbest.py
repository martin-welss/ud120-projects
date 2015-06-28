#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit

features_list = ['poi','from_this_person_to_poi', 'total_stock_value','total_payments','expenses', 'other', 'restricted_stock','salary', 'bonus'] # You will need to use more features
#features_list = ['poi', 'bonus'] # You will need to use more features


### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )

### Task 2: Remove outliers
data_dict.pop('TOTAL',0)

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
features_list.append('email_fraction_from_poi')

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

from sklearn.feature_selection import SelectKBest

for f in range(1,11): 
    kbest = SelectKBest(k=f).fit(features, labels)
    feature_names = [features_list[i+1] for i in kbest.get_support(indices=True)]
    print feature_names
