#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
    
# Initialize transformers
transformers = [
    deepchem.trans.NormalizationTransformer(
    transform_y=True, dataset=train_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
]
for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)
transformers = [
    deepchem.trans.NormalizationTransformer(
    transform_y=True, dataset=test_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
]
for transformer in transformers:
    test_dataset = transformer.transform(test_dataset)

# define splitter for cross validation    
splitter = deepchem.splits.RandomSplitter()
#train, valid, test = splitter.train_valid_test_split(train_dataset) 
split_datas = splitter.k_fold_split(train_dataset,5)

# define the RMSE and P2 metric
metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
metric_r2 = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)

# conduct cross-validation
validation_scores = []
train_scores = []
for train_set, val_set in split_datas:
    model = dc.models.GraphConvModel(len(flashpoint_tasks),dense_layer_size = 512, mode='regression')
    model.fit(train_set,nb_epoch= 70,batch_size = 5)
    train_score = model.evaluate(train_set, [metric], transformers)
    train_scores.append(list(train_score.values()).pop()) 
    valid_score = model.evaluate(val_set, [metric], transformers)
    validation_scores.append(list(valid_score.values()).pop())    
    
# print out the results of cross validation    
print("=========== Results ===========")
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
print("cross_validation_score: ")
print(cross_validation_score)

# train the model with the whole traning set and get the test score
model = dc.models.GraphConvModel(len(flashpoint_tasks), dense_layer_size = 128, mode='regression')
model.fit(train_dataset, nb_epoch= 100,batch_size = 64)
print("cross_test_score(RMSE): ")
test_score = model.evaluate(test_dataset, [metric], transformers),
print("cross_test_score(p2): ")
test_score_r2 = model.evaluate(test_dataset, [metric_r2], transformers)

# plot the parity plot over test dataset
y_true = test_dataset.y
y_pred = model.predict(test_dataset)
y_true = undo_transforms(y_true, transformers)  # unnormlaize the result for plotting the parity plot
#print("------------y_true--------------")
#print(y_true)
y_pred = undo_transforms(y_pred, transformers)  # unnormlaize the result for plotting the parity plot
#print("------------y_pred--------------")
#print(y_pred)
my_path = "/u/x/i/xiaoyus/private/Skunkworks"
img_name = "Carroll_plot.png"
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

