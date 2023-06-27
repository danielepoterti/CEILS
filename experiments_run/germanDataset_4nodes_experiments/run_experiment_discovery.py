import sys
sys.path.append('C:/Users/danie/OneDrive/Desktop/development/CEILS')

import pandas as pd
import numpy as np
from core.causal.build_causal_skeleton import *
from causallearn.utils.GraphUtils import GraphUtils



def load_germandataset(nodes):
    '''
    read dataset and preprocessing for german credit dataset
    return data only for the nodes
    '''
    
    df = pd.read_csv("C:/Users/danie/OneDrive/Desktop/development/CEILS/experiments_run/germanDataset_4nodes_experiments/data/german_data_credit_dataset.csv")
    #create quickaccess list with categorical variables labels
    catvars = ['existingchecking', 'credithistory', 'purpose', 'savings', 'employmentsince',
               'statussex', 'otherdebtors', 'property', 'otherinstallmentplans', 'housing', 'job',
               'telephone', 'foreignworker']
    #create quickaccess list with numerical variables labels
    numvars = ['creditamount', 'duration', 'installmentrate', 'residencesince', 'age',
               'existingcredits', 'peopleliable', 'classification']

    # Binarize the target 0 = 'bad' credit; 1 = 'good' credit
    df.classification.replace([1,2], [1,0], inplace=True)


    #  dic categories Index(['A11', 'A12', 'A13', 'A14'], dtype='object')
    dict_categorical = {}
    for c in catvars:
        dict_categorical[c] = list(df[c].astype("category").cat.categories)
        df[c] = df[c].astype("category").cat.codes

    #  create gender variable 1= female 0 = male

    df.loc[df["statussex"] == 0, "gender"] = 0
    df.loc[df["statussex"] == 1, "gender"] = 1
    df.loc[df["statussex"] == 2, "gender"] = 0
    df.loc[df["statussex"] == 3, "gender"] = 0
    df.loc[df["statussex"] == 4, "gender"] = 1

    #  all features as float
    df = df.astype("float64")
    df["classification"] = df["classification"].astype("int32")
    # save codes
    with open('dict_german.txt', 'w') as f:
        f.write(str(dict_categorical))

    return df[nodes]

df = load_germandataset(["gender", "age", "creditamount", "classification", "duration"])

print(build_causal_skeleton(df))
