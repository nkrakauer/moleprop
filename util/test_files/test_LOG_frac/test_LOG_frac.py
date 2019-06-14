import sys
sys.path.append('../')     # change the path to your dir of workflow.py
import workflow as wf
import pandas as pd

print("About to load")
loader = wf.Loader
data = loader.load(file_name = '9k_dataset.csv',data_dir = '/srv/home/xsun256/Moleprop/summer19/datasets')

print("About to split")
splitter = wf.Splitter
leave_out_group = 'pubchem'  #TODO: change it to the group you want to leave out
fraction = 0.8               #TODO: change it to the fraction of left-out group you want to use for test set
print(str(fraction) + " of "+leave_out_group + " will be used as test set")
indices,dataset = splitter.LOG(data, test_group = leave_out_group, frac = fraction)
train,test = indices

'''
args = {'nb_epoch': 80,
        'batch_size': 50,
        'n_tasks': 1,
        'graph_conv_layers':[64,64],
        'dense_layer_size': 256,
#        'dropout': 0.0,           # for testing if this workflow tool can correctly use default dropout if it is not inputted
        'mode': 'regression'}
'''

args = None
print("About to conduct LOG_validation")
scores,prediction,test_dataset = wf.Run.LOG_validation(dataset,indices, model = 'GC',model_args = args,metrics = ['AAD', 'RMSE', 'MAE', 'R2'])

for key in scores:
    print(key+" = "+str(scores[key]))

print("About to plot full data")
txt = {'RMSE/STD': scores['RMSE']/test_dataset['flashpoint'].std(),
       'RMSE': scores['RMSE'],
       'MAE': scores['MAE'],
       'R2': scores['R2'],
       'AAD': scores['AAD']}
wf.Plotter.parity_plot(prediction,test_dataset,plot_name = "LOG_puchem_parity", text = txt)
wf.Plotter.residual_histogram(prediction,test_dataset,plot_name = "LOG_pubchem_residual", text = txt)

