#!/usr/bin/env python
# coding: utf-8

# ## New features (updated 6/13/2019):
#     1. implementation for integrated plot
#     2. workflow tool can automatically create folders for plots and data info now
#     3. more well-structured output files

# # STEP 1
# - import workflow.py and need to make sure that integration_helpers.py is under the same dir as workflow.py<br/>
# - change '../' to your dir of workflow.py

# In[3]:


import sys
sys.path.append('../')     
import workflow as wf
import pandas as pd


# # STEP 2
# - Loading raw data(dataset with duplicates) from the path of "data_dir + file_name" <br/>
# - Change data_dir and file_name to the path and file you want to run

# In[ ]:


print("About to load")
loader = wf.Loader
data = loader.load(file_name = 'integrated_dataset.csv',data_dir = '/srv/home/xsun256/Moleprop/summer19')


# # STEP 3
# - Data cleaning (removing duplictaes) and splitting
# - Returns indices of training sets and test sets, and new dataset after removing duplicates
# - params:
#   - data: DataFrame
#   - n_splits (for k_fold splitter): n of n-fold cv
#   - test_group (for LOG splitter): name of the dataset you want to leave out as test group
# - Other params you can set:
#   - k_fold splitter:
#     - shuffle( = True in defualt): True or False
#   - LOG splitter: 
#     - frac( = 1 in default): 
#         - training set: all other datasets + (1-frac) * left-out dataset
#         - test set: frac* left-out dataset

# In[ ]:


print("About to split")
splitter = wf.Splitter
indices,dataset = splitter.k_fold(data, n_splits = 3)
# indices,dataset = splitter.LOG(data,'left-out group name')  # LOG splitter


# # STEP 4
# - conduct CV or LOG_validation
# - Param:
#     - dataset: DataFrame
#     - indices: indices returned from splitter
#     - model name: accept "GC","GraphConv","graphconv", and "MPNN"
#     - model_args (= None in default): 
#         - GC args:
#             - nb_epoch, 
#             - batch_size, 
#             - n_tasks, 
#             - graph_conv_layers, 
#             - dense_layer_size, 
#             - dropout, 
#             - mode
#         - MPNN args:
#             - 'n_tasks':1,
#             - 'n_atom_feat':75,
#             - 'n_pair_feat':14,      # NEED to be 14 for WaveFeaturizer
#             - 'T':1,
#             - 'M':1,
#             - 'batch_size':32,
#             - 'nb_epoch': 50,
#             - 'learning_rate':0.0001,
#             - 'use_queue':False,
#             - 'mode':"regression"
#      - metrics: accept "AAD", "RMSE", "MAE", "R2"
# - Return:
#     - scores: a dictionart of scores (based on the merics you set):
#         - avg_AAD and AAD_list (list of AAD score from every iteration)
#         - avg_R2 and R2_list
#         - avg_RMSE and RMSE_list
#         - avg_MAE and MAE_list
#     - predictions: list of predictions from every iteration
#     - test_Dataset: list of test datasets from every iteration

# In[1]:


'''
GC_args_example = {'nb_epoch': 80,
        'batch_size': 50,
        'n_tasks': 1,
        'graph_conv_layers':[64,64],
        'dense_layer_size': 256,
        'dropout': 0.0,           
        'mode': 'regression'}
'''
print("About to simulate")
# Cross Validation
scores,predictions,test_datasets = wf.Run.cv(dataset,indices, 
                                             'GC',
                                             model_args = None,
                                             n_splits = 3, 
                                             metrics = ['AAD', 'RMSE', 'MAE', 'R2'])
# LOG validation
'''
scores,predictions,test_datasets = wf.Run.cv(dataset,indices, 
                                             'GC',
                                             model_args = None,
                                             metrics = ['AAD', 'RMSE', 'MAE', 'R2'])
'''


# # STEP5
# - Print out result and save parity plots and hitogram plots
# - Params:
#     - plot_name: name of plot without add '.png'
#     - text: **dictionary** of scores that you want to add to plot

# In[ ]:


# Make plots for every fold
for key in scores:
    print(key+" = "+str(scores[key]))

print("About to make parity plots")
for i in range(len(predictions)):
    p_name = "parity_"+str(i)
    std = test_datasets[i]['flashpoint'].std()
    txt = {
           "RMSE":scores['RMSE_list'][i], 
           "R2":scores['R2_list'][i], 
           "MAE":scores['MAE_list'][i], 
           "AAD":scores['AAD_list'][i],
           "RMSE/std":scores['RMSE_list'][i]/std}
    wf.Plotter.parity_plot(predictions[i],test_datasets[i], plot_name = p_name, text = txt)

print("About to make residual plot")
for i in range(len(predictions)):
    r_name = "residual_"+str(i)
    std = test_datasets[i]['flashpoint'].std()
    txt = {
           "RMSE":scores['RMSE_list'][i],
           "R2":scores['R2_list'][i],
           "MAE":scores['MAE_list'][i],
           "AAD":scores['AAD_list'][i],
           "RMSE/std":scores['RMSE_list'][i]/std}
    wf.Plotter.residual_histogram(predictions[i],test_datasets[i], plot_name = r_name, text = txt)

# For making integrated plots 
print("About to plot full data")
P = predictions[0] + predictions[1]+ predictions[2]
D = pd.concat(test_datasets)
txt = {'RMSE/STD': scores['RMSE']/D['flashpoint'].std(),
       'RMSE': scores['RMSE'],
       'MAE': scores['MAE'],
       'R2': scores['R2'],
       'AAD': scores['AAD']}
wf.Plotter.parity_plot(P,D,plot_name = "Full_parity", text = txt)
wf.Plotter.residual_histogram(P,D,plot_name = "Full_residual", text = txt)

