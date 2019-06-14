import sys
sys.path.append('../')     # change the path to your dir of workflow.py
import workflow as wf
import pandas as pd
import statistics as stat

print("About to load")
loader = wf.Loader
# TODO:need to change file name and path to your local file
data = loader.load(file_name = 'sample.csv',data_dir = '/srv/home/xsun256/Moleprop/summer19/datasets')

multi_predictions = list() # list of lists of lists
multi_scores = list()      # for getting the average scores 
repetition = 2             # number of cv we want to conduct

print("About to split")
splitter = wf.Splitter
# TODO: change test_group to the group name you want to leave out as test set
ind,dataset = splitter.LOG(data, test_group = 'pubchem')
leave_out_group = pd.DataFrame()

# conduct multiple cross validation
for r in range(repetition):

    '''
    args = {'nb_epoch': 80,
        'batch_size': 50,
        'n_tasks': 1,
        'graph_conv_layers':[64,64],
        'dense_layer_size': 256,
        'dropout': 0.0,           
        'mode': 'regression'}
    '''

    args = None
    print("About to simulate")
    scores,pred,test_dataset = wf.Run.LOG_validation(dataset,ind, model ='GC',model_args = args, metrics = ['AAD', 'RMSE', 'MAE', 'R2'])

    for key in scores:
        print(key+" = "+str(scores[key]))

    print("About to make parity/residual plots")
    txt = {'RMSE/STD': scores['RMSE']/test_dataset['flashpoint'].std(),
           'RMSE': scores['RMSE'],
           'MAE': scores['MAE'],
           'R2': scores['R2'],
           'AAD': scores['AAD']}
    wf.Plotter.parity_plot(pred,test_dataset,plot_name = str(r)+"_LOG_parity", text = txt)
    wf.Plotter.residual_histogram(pred,test_dataset,plot_name = str(r)+"_LOG_residual", text = txt)

    # store scores for final result and plots
    m_scores = {'RMSE':scores['RMSE'], 'MAE':scores['MAE'], 'R2':scores['R2'], 'AAD':scores['AAD']}
    multi_scores.append(m_scores)
    multi_predictions.append(pred)
    leave_out_group = test_dataset

# get the final result and plot the final errorbar-parity plot
f_scores = {'RMSE':0, 'MAE':0, 'R2':0, 'AAD':0}
f_pred = [[] for x in range(len(multi_predictions[0]))]

for i in range(repetition):
    for key in f_scores:
        f_scores[key]+=multi_scores[i][key]/repetition
    for k in range(len(multi_predictions[i])):
        f_pred[k].append(multi_predictions[i][k])

f_scores['RMSE/STD'] = f_scores['RMSE']/leave_out_group['flashpoint'].std()

wf.Plotter.parity_plot(f_pred,leave_out_group,plot_name = "FINAL_Full_parity", text = f_scores, errorbar = True)
f_avg_pred = list()
for i in range(len(f_pred)):
    f_avg_pred.append(stat.mean(f_pred[i]))
wf.Plotter.residual_histogram(f_avg_pred,leave_out_group,plot_name = "FINAL_Full_residual", text = f_scores)
