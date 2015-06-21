#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop('TOTAL',0)
features = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses',
'from_poi_to_this_person', 'exercised_stock_options', 'from_messages', 'other',
'from_this_person_to_poi', 'long_term_incentive', 'shared_receipt_with_poi', 'restricted_stock',
'director_fees']

data = featureFormat(data_dict, features)


def plot_scatter(feature1, feature2):
    for point in data:
        xfeature = point[features.index(feature1)]
        yfeature = point[features.index(feature2)]
        matplotlib.pyplot.scatter( xfeature, yfeature )

    matplotlib.pyplot.xlabel(feature1)
    matplotlib.pyplot.ylabel(feature2)
    matplotlib.pyplot.show()

def dump_feature(feature):
    nan=0
    for person in data_dict:
        if data_dict[person][feature]=="NaN":
            nan+=1
        print person, '\t\t', data_dict[person][feature]
    print feature,"NaN count: ",nan


def plot_hist(feature):
    xfeature=[]
    for point in data:
        xfeature.append(point[features.index(feature)])

    matplotlib.pyplot.hist(xfeature, 50)
    matplotlib.pyplot.xlabel(feature)
    matplotlib.pyplot.ylabel("count")
    matplotlib.pyplot.show()
    
