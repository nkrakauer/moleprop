import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# import pkg for model

def load(file_name, data_dir = './'):
# make it a loader class which can return some statistic of this dataset?
    """
    load data from .csv file
    """
    # TODO: need to clean data before loading?
    data_file = os.path.join(data_dir, file_name)
    if not os.path.exists(data_file):
        if data_dir == './':
            data_dir = "current"
        error_msg = file_name + " was not found in " + data_dir + " directory"
        sys.exit(error_msg)
    print("|||||||||||||||||||||Loading " + file_name+ "|||||||||||||||||||||||")
    data = pd.read_csv(data_file) # encoding='latin-1' might be needed
    return data

def k_fold_splitter(dataset, n_splits = 3):
    """
    split data into k-fold
    return indices of training and test sets
    """
    kf = KFold(n_splits)
    indices = kf.split(dataset)
    return indices

def cv(file_name, model, data_dir = './', n_splits = 3):   
    """
    pass data into models (MPNN or GraphConv) and conduct cross validation
    """
    if model != 'MPNN' or model != 'GraphConv':
        sys.exit("Only support MPNN model and GraphConv model")
    data = load(file_name, data_dir)
    indice = k_fold_splitter(data,n_splits)
    cv_scores = []
    for train_indice, test_indice in indice:
        train_set = data.iloc[train_indice]
        test_set = data.iloc[test_indice]
        if model == 'MPNN':
            ### define MPNN model
            ### model.fit(train_set)
            ### score = model.evaluate(test_set)
        else if model == 'GraphConv':
            ### define GraphConv model
            ### model.fit(train_set)
            ### score = model.evaluate(test_set)        
        cv_scores.append(score)
    avg_cv_score = sum(cv_scores)/n_splits
    return avg_cv_score
