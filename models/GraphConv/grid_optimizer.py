#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
__author__ = "Bharath Ramsundar"
__copyright__ = "Copyright 2016, Stanford University"
__license__ = "MIT"

import os
import unittest
import tempfile
import shutil
import numpy as np
import tensorflow as tf
import deepchem as dc
import logging
import math
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel
'''
# load flashpoint data set
flashpoint_tasks, flashpoint_datasets, transformers = dc.molnet.load_flashpoint(
    "/u/x/i/xiaoyus/private/Skunkworks",
    featurizer='GraphConv')
train_dataset, valid_dataset, test_dataset = flashpoint_datasets
'''
data_dir = "path/to/dataset"  
train_dataset_file = os.path.join(data_dir, "dataset_name.csv")
if not os.path.exists(train_dataset_file):
    print("data set was not found in the given directory")
flashpoint_tasks = ['flashPoint']  # column name need to be exactly 'flashPoint'
loader = dc.data.CSVLoader(
    tasks=flashpoint_tasks, smiles_field="smiles", featurizer = dc.feat.ConvMolFeaturizer())
train_dataset = loader.featurize(train_dataset_file, shard_size=8192)
    
# Initialize transformers
transformers = [
    dc.trans.NormalizationTransformer(
    transform_y=True, dataset=train_dataset, move_mean=True)
]
for transformer in transformers:
    train_dataset = transformer.transform(train_dataset)

# define splitter for cross validation    
splitter = dc.splits.RandomSplitter()
train_set, valid_set = splitter.train_test_split(train_dataset) 

# Define metric for eavluating the model by using Pearson_R2
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
# Define metric for eavluating the model by using RMSE
#metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)

params_dict = {
    "nb_epoch":[10,50,100],
    "batch_size": [8,32,64],
    "n_tasks":[1],
    "graph_conv_layers":[[64,64]],#[32, 32],[64],[128, 128], [64,64,64]],
    "dense_layer_size":[128,256,512],
    "dropout":[0.4,0.5],
    "mode":["regression"],
    "number_atom_features":[75]
}

def gc_model_builder(model_params , model_dir): 
    gc_model = GraphConvModel(**model_params, model_dir = "path/to/save/models") 
    return gc_model

optimizer = dc.hyper.HyperparamOpt(gc_model_builder)
best_model, best_hyperparams, all_results = optimizer.hyperparam_search(
    params_dict,
    train_set,
    valid_set,
    transformers,
    metric,
    logdir=None)
#    use_max=False)  # setting use_max to be False when using RMSE as metric


print("\n===================BEST MODEL=================")
print(best_model)
print("\n===================BEST Params=================")
print(best_hyperparams)
print("\n===================ALL_RESULTS=================")

'''
# print out all resutls 
for key in sorted(all_results.keys()):
    print(key,": ", round(all_results[key],4))
'''

