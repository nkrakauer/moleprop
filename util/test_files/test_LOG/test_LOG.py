import sys
<<<<<<< Updated upstream
sys.path.append('/srv/home/apolitowicz/moleprop/util/')
=======
sys.path.append('/srv/home/nkrakauer/moleprop/util')
>>>>>>> Stashed changes
import workflow as wf
import pandas as pd

print("About to load")
loader = wf.Loader
# TODO: need to change name and dir to your local dataset name and path
<<<<<<< Updated upstream
data = loader.load(file_name = 'new_integrated_dataset.csv',data_dir = '/srv/home/apolitowicz/moleprop/data')

=======
data = loader.load(file_name = 'integrated_dataset.csv',data_dir = '/srv/home/nkrakauer/moleprop/data/')
group = 'pan07'
>>>>>>> Stashed changes
print("About to split")
splitter = wf.Splitter
# TODO: change test_group to the group name you want to leave out
indices,dataset = splitter.LOG(data, test_group = group, frac=1/3)

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

# printing out scores
for key in scores:
    print(key+" = "+str(scores[key]))

# getting plots
print("About to plot full data")
txt = {'RMSE/STD': scores['RMSE']/test_dataset['flashpoint'].std(),
       'RMSE': scores['RMSE'],
       'MAE': scores['MAE'],
       'R2': scores['R2'],
       'AAD': scores['AAD']}
wf.Plotter.parity_plot(prediction,test_dataset,plot_name = group+"_parity", text = txt)
wf.Plotter.residual_histogram(prediction,test_dataset,plot_name = group+"_residual", text = txt)
