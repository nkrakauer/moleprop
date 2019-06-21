#!/usr/bin/env python

import sys
sys.path.append('/srv/home/apolitowicz/moleprop/util/')
import workflow as wf
import pandas as pd


print("About to load")
loader = wf.Loader
data = loader.load(file_name = 'integrated_dataset_grouped.csv',data_dir = '/srv/home/apolitowicz/moleprop/data')
#data = loader.load(file_name = 'geleste_complete.csv',data_dir = '/srv/home/apolitowicz/moleprop/data')

print("About to split")
splitter = wf.Splitter
#indices,dataset = splitter.k_fold(data, n_splits = 10)
#indices,dataset = splitter.LOG(data,'gelest_germanium',1)  # LOG splitter
indices,dataset = splitter.basic_transfer_splits(data,'gelest_germanium', True, False)
#silicon_dataset = splitter.get_organosilicons(data)
#metallic_dataset = splitter.get_organometallics(data)

# DEBUG
#print(silicon_dataset.shape)
#silicon_dataset.to_csv("silicons.csv")
#print(metallic_dataset.shape)
#metallic_dataset.to_csv("metallics.csv")

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
#scores,predictions,test_datasets = wf.Run.cv(dataset,indices, 
#                                             'GC',
#                                             model_args = None,
#                                             n_splits = 10, 
#                                             metrics = ['AAD', 'RMSE', 'MAE', 'R2'])

# LOG validation
#scores,predictions,test_datasets = wf.Run.LOG_validation(dataset,indices, 
#                                             'GC',
#                                             model_args = None,
#                                             metrics = ['AAD', 'RMSE', 'MAE', 'R2'])

# basic transfer learning
scores,predictions,test_datasets = wf.Run.basic_transfer(dataset,indices, 
                                             'GC',
                                             model_args = None,
                                             metrics = ['AAD', 'RMSE', 'MAE', 'R2'],
                                             nn_edit = False)

# Make plots for every fold
for key in scores:
    print(key+" = "+str(scores[key]))

# DEBUG
print("[DEBUG] len predictions: ", len(predictions))

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
#P = predictions[0] + predictions[1]+ predictions[2]
P = []
for p in predictions:
    P = P + p
D = pd.concat(test_datasets)
txt = {'RMSE/STD': scores['RMSE']/D['flashpoint'].std(),
       'RMSE': scores['RMSE'],
       'MAE': scores['MAE'],
       'R2': scores['R2'],
       'AAD': scores['AAD']}
wf.Plotter.parity_plot(P,D,plot_name = "Full_parity", text = txt)
wf.Plotter.residual_histogram(P,D,plot_name = "Full_residual", text = txt)
