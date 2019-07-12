import sys
sys.path.append('../../util/')     # change the path to your dir of workflow.py
import workflow as wf
import pandas as pd

print("About to load")
loader = wf.Loader
# TODO need to change file name and dir to your local environment
data = loader.load(file_name = 'carroll11.csv',data_dir = '/srv/home/nkrakauer/moleprop/data')

print("About to split")
splitter = wf.Splitter
indices,dataset = splitter.k_fold(data, n_splits = 5)

args = {'nb_epoch': 150,
        'learning_rate':0.001,
        'batch_size': 8,
        'n_tasks': 1,
        'graph_conv_layers':[64,64],
        'dense_layer_size': 512,
#        'dropout': 0.0,           # for testing if this workflow tool can correctly use default dropout if it is not inputted
        'mode': 'regression'}

mpnnargs = {'nb_epoch': 250,
            'learning_rate':0.0005,
            'batch_size':32,
            'n_tasks':1,
            'n_atom_feat':75,
            'n_pair_feat':14,
            'T':1,
            'M':1,
            'dropout':0.2,
            'mode':'regression'
        }

print("About to conduct cross validation")
scores,predictions,test_datasets = wf.Run.cv(dataset,indices, model = 'MPNN',model_args = None,n_splits = 5, metrics = ['train','AAD', 'RMSE', 'MAE', 'R2'])

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

print("About to plot full data")
P = list()                               # integration of predictions for whole dataset
for i in range(len(predictions)):
    P += predictions[i]
D = pd.concat(test_datasets)             # integration of the whole dataset
txt = {'RMSE/STD': scores['RMSE']/D['flashpoint'].std(),
       'RMSE': scores['RMSE'],
       'MAE': scores['MAE'],
       'R2': scores['R2'],
       'AAD': scores['AAD']}
wf.Plotter.parity_plot(P,D,plot_name = "Full_parity", text = txt)
wf.Plotter.residual_histogram(P,D,plot_name = "Full_residual", text = txt)
