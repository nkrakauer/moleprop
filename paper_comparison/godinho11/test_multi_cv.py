import sys
sys.path.append('../../moleprop/util')     # change the path to your dir of workflow.py
import workflow as wf
import pandas as pd
import statistics as stat

print("About to load")
loader = wf.Loader
data = loader.load(file_name = 'godinho11.csv',data_dir = '/srv/home/xsun256/moleprop/data')

multi_predictions = list() # list of lists of lists
multi_scores = list()      # for getting the average scores 
repetition = 5             # number of cv we want to conduct

print("About to split")
splitter = wf.Splitter
ind,dataset = splitter.k_fold(data, n_splits = 5)

indices = list()
for train,test in ind:
    indices.append((train,test))

# conduct multiple cross validation
for r in range(repetition):

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
    print("About to simulate")
    scores,predictions,test_datasets = wf.Run.cv(dataset,indices, 'GC',model_args = args,n_splits = 5, metrics = ['AAD', 'RMSE', 'MAE', 'R2'])

    for key in scores:
        print(key+" = "+str(scores[key]))

    print("About to make parity plots")
    for i in range(len(predictions)):
        p_name = str(r)+"parity_cv"+str(i)
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
        r_name = str(r)+"_residual_cv"+str(i)
        std = test_datasets[i]['flashpoint'].std()
        txt = {
               "RMSE":scores['RMSE_list'][i],
               "R2":scores['R2_list'][i],
               "MAE":scores['MAE_list'][i],
               "AAD":scores['AAD_list'][i],
               "RMSE/std":scores['RMSE_list'][i]/std}
        wf.Plotter.residual_histogram(predictions[i],test_datasets[i], plot_name = r_name, text = txt)

    print("About to plot full data")
    P = list()
    for i in range(5): 
        P += predictions[i]   
    D = pd.concat(test_datasets)
    txt = {'RMSE/STD': scores['RMSE']/D['flashpoint'].std(),
           'RMSE': scores['RMSE'],
           'MAE': scores['MAE'],
           'R2': scores['R2'],
           'AAD': scores['AAD']}
    wf.Plotter.parity_plot(P,D,plot_name = str(r)+"_Full_parity", text = txt)
    wf.Plotter.residual_histogram(P,D,plot_name = str(r)+"_Full_residual", text = txt)

    # store scores for final result and plots
    m_scores = {'RMSE':scores['RMSE'], 'MAE':scores['MAE'], 'R2':scores['R2'], 'AAD':scores['AAD']}
    multi_scores.append(m_scores)
    multi_predictions.append(P)

# get the final result and plot the final error bar parity_plot
f_scores = {'RMSE':0, 'MAE':0, 'R2':0, 'AAD':0}
f_pred = [[] for x in range(len(multi_predictions[0]))]

for i in range(repetition):
    for key in f_scores:
        f_scores[key]+=multi_scores[i][key]/repetition
    for k in range(len(multi_predictions[i])):
        f_pred[k].append(multi_predictions[i][k])

f_scores['RMSE/STD'] = f_scores['RMSE']/dataset['flashpoint'].std()

wf.Plotter.parity_plot(f_pred,D,plot_name = "FINAL_Full_parity", text = f_scores, errorbar = True)
f_avg_pred = list()
for i in range(len(f_pred)):
    f_avg_pred.append(stat.mean(f_pred[i]))
wf.Plotter.residual_histogram(f_avg_pred,D,plot_name = "FINAL_Full_residual", text = f_scores)
