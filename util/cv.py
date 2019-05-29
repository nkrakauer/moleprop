import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
# import pkg for model

def load(file_name, data_dir = './'):
    """
    load data from .csv file
    """
    # TODO: need to clean data before loading?
    data_file = os.path.join(data_dir, file_name)
    if not os.path.exists(data_file):
        sys.exit(file_name + " was not found in " + data_dir + " directory")
    print("|||||||||||||||||||||Loading " + file_name+ "|||||||||||||||||||||||")
    data = pd.read_csv(data_file) # encoding='latin-1' might be needed
    return data
  # make it a loader class which can return some statistic of this dataset?
  # transform data if needed?

def k_fold_splitter(dataset, n_splits = 3):
    """
    split data into k-fold
    return indices of training and test sets
    """
    kf = KFold(n_splits)
    indices = kf.split(dataset)
    return indices
    
def cv(file_name, data_dir = './', n_splits = 3):   
    """
    pass data into models (MPNN or GraphConv) and conduct cross validation
    """
    data = load(file_name, data_dir)
    indice = k_fold_splitter(data,n_splits)
    cv_scores = []
    for train_indices, test_indices in indice:
        train_set = data.iloc[train_indices]
        test_set = data.iloc[test_indices]
        ### define model
        ### model.fit(train_set)
        ### score = model.evaluate(test_set)
        cv_scores.append(score)
    avg_cv_score = sum(cv_scores)/n_splits
    return avg_cv_score
