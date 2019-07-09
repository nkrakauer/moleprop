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
from deepchem.models import MPNNModel

model = 'MPNN'
dataset = 'carroll10'
data_dir = "../../data/"  
dataset_file = os.path.join(data_dir, dataset)
if not os.path.exists(dataset_file):
    print("data set was not found in the given directory")

flashpoint_tasks = ['flashpoint']
if model == 'GCNN':
    loader = dc.data.CSVLoader(tasks=flashpoint_tasks, smiles_field="smiles",
            featurizer = dc.feat.ConvMolFeaturizer())
else:
    loader = dc.data.CSVLoader(tasks=flashpoint_tasks, smiles_field='smiles',
            featurizer = dc.feat.WeaveFeaturizer())

dataset = loader.featurize(dataset_file, shard_size=8192)

# Initialize transformers
transformers = [
    dc.trans.NormalizationTransformer(
    transform_y=True, dataset=dataset, move_mean=True)
]
for transformer in transformers:
    train_dataset = transformer.transform(dataset)

# define splitter for cross validation    
splitter = dc.splits.RandomSplitter()
train_set, valid_set = splitter.train_test_split(dataset, frac_train=.5) 

# Define metric for eavluating the model by using Pearson_R2
metric = dc.metrics.Metric(dc.metrics.pearson_r2_score, np.mean)
# Define metric for eavluating the model by using RMSE
#metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
if model == 'GCNN':
    params_dict = {
        "nb_epoch":[200],#, 200, 300, 400],
        "batch_size": [8,32,64,128,256],
        "n_tasks":[1],
        "graph_conv_layers":[[64,64]],#,[32, 32],[64],[128, 128], [64,64,64]],
        "dense_layer_size":[256],
        "dropout":[0.0],
        "mode":["regression"],
        "number_atom_features":[75]
    }
else:
    params_dict = {
            "nb_epoch": [70, 100, 150, 250, 400],
            "batch_size": [8,32],
            "n_tasks": [1],
            "n_atom_feat": [75],
            "n_pair_feat": [14],
            "T": [1],
            "M": [1],
            "queue": [False],
            "dropout": [0.0, 0.2, 0.4],
            "learning_rate": [0.0005,0.001,0.005]
    }

def gc_model_builder(model_params, model_dir): 
    gc_model = GraphConvModel(**model_params, model_dir = "./epoch_models") 
    return gc_model

def mpnn_model_builder(model_params, model_dir):
    mpnn_model = dc.models.MPNNModel(**model_params, model_dir = "./epoch_models")
    return mpnn_model

optimizer = dc.hyper.HyperparamOpt(mpnn_model_builder)
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

# print out all resutls 
for key in sorted(all_results.keys()):
    print(key,": ", round(all_results[key],4))
