from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import logging
import deepchem
import math
import numpy as np
import tensorflow as tf
import deepchem as dc
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
from deepchem.trans.transformers import undo_transforms
import matplotlib.pyplot as plt
#plt.switch_backend('agg') #used for ssh plotting

# Load dataset
print("||||||||||||||||||||||Loading Dataset|||||||||||||||||||")
data_dir = "path/to/dataset"  #Hardcoded dir_address, need to be modified
train_dataset_file = os.path.join(data_dir, "train_dataset_name.csv")
test_dataset_file = os.path.join(data_dir, "test_dataset_name.csv")
if not os.path.exists(train_dataset_file) or not os.path.exists(test_dataset_file):
    print("data set was not found in the given directory")
flashpoint_tasks = ['flashPoint']  # Need to set the column name to be excatly "flashPoint"
loader = deepchem.data.CSVLoader(
    tasks=flashpoint_tasks, smiles_field="smiles", featurizer = deepchem.feat.ConvMolFeaturizer())
train_dataset = loader.featurize(train_dataset_file, shard_size=8192)
test_dataset = loader.featurize(test_dataset_file, shard_size=8192)

# define splitter for cross validation
splitter = deepchem.splits.RandomSplitter()
split_datas = splitter.k_fold_split(train_dataset,5)


# define the RMSE and P2 metric
metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# conduct cross-validation
validation_scores = []
train_scores = []
for train_set, val_set in split_datas:
    # initialize cv transformers
    cv_train_transformers = [
        deepchem.trans.NormalizationTransformer(
        transform_y=True, dataset=train_set)
    ]
    for transformer in cv_train_transformers:
        train_set = transformer.transform(train_set)
    cv_val_transformers = [
        deepchem.trans.NormalizationTransformer(
        transform_y=True, dataset= val_set) 
    ]
    for transformer in cv_val_transformers:
        val_set = transformer.transform(val_set)
    model = dc.models.GraphConvModel(len(flashpoint_tasks),batch_size = 50, mode='regression')
    model.fit(train_set, nb_epoch = 125)
    train_score = model.evaluate(train_set, [metric], cv_train_transformers)
    train_scores.append(list(train_score.values()).pop())
    valid_score = model.evaluate(val_set, [metric], cv_val_transformers)
    validation_scores.append(list(valid_score.values()).pop())    
    
# print out the results of cross validation
print("===========Cross Validation Results===========")
cross_train_score = 0
for val in train_scores:
    cross_train_score += val
cross_train_score = cross_train_score/5
print("cross_train_score: ")
print(cross_train_score)
cross_validation_score = 0
for val in validation_scores:
    cross_validation_score += val
cross_validation_score = cross_validation_score/5
print("cross_test_score: ")
print(cross_validation_score)

# Initialize train transformers
train_transformers = [
    deepchem.trans.NormalizationTransformer(
    transform_y=True, dataset=train_dataset) 
]
for transformer in train_transformers:
    train_dataset = transformer.transform(train_dataset)

# Initialize test transformers
test_transformers = [
    deepchem.trans.NormalizationTransformer(
    transform_y=True, dataset=test_dataset)
]
for transformer in test_transformers:
    test_dataset = transformer.transform(test_dataset)
    
# train the model with the whole traning set and get the test score
model = dc.models.GraphConvModel(len(flashpoint_tasks),batch_size = 64, mode='regression')
model.fit(train_dataset, nb_epoch = 125)
print("cross_test_score(RMSE): ")
test_score = model.evaluate(test_dataset, [metric], test_transformers)
print("cross_test_score(p2): ")
test_score_r2 = model.evaluate(test_dataset, [metric_r2], test_transformers)

'''
# plot the parity plot over test dataset
y_true = test_dataset.y
y_pred = model.predict(test_dataset)
y_true = undo_transforms(y_true, test_transformers)  # unnormlaize the result for plotting the parity plot
# print("------------y_true--------------")
# print(y_true)
y_pred = undo_transforms(y_pred, test_transformers)  # unnormlaize the result for plotting the parity plot
# print("------------y_pred--------------")
# print(y_pred)

# plot the parity plot over test dataset
y_true = test_dataset.y
y_pred = model.predict(test_dataset)
y_true = undo_transforms(y_true, test_transformers)  # unnormlaize the result for plotting the parity plot
#print("------------y_true--------------")
#print(y_true)
y_pred = undo_transforms(y_pred, test_transformers)  # unnormlaize the result for plotting the parity plot
#print("------------y_pred--------------")
#print(y_pred)
my_path = "path/to/save/plot"
img_name = "plot_name.png"
score_rmse = "RMSE = " + str(round(list(test_score.values()).pop(),2))
score_r2 = "R^2 score = " + str(round(list(test_score_r2.values()).pop(),2))
#plt.figure(dpi=1300)  # higher resolution for the image
#plt.rcParams.update({'font.size': 13}) # for changing plot font size
plt.title("Carroll Parity Plot")
plt.xlabel("Experimental (Kelvin)")
plt.ylabel("Predicted (Kelvin)")   
#plt.grid(True)
plt.scatter(y_true,y_pred, label='flash point')
plt.text(350,290, score_rmse)
plt.text(350,270, score_r2)
#plt.text('upper left','P2 = %s', list(test_score_p2).pop())
plt.plot([200,450],[200,450],'k-',label = 'best fit')
plt.legend(loc='lower right')
#plt.savefig(os.path.join(my_path,img_name))
plt.show()
#plt.savefig('Parity Plot Carroll test_dataset.png')
'''
