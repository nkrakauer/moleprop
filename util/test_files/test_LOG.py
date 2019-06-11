import workflow as wf
import pandas as pd

print("About to load")
loader = wf.Loader
data = loader.load(file_name = 'integrated_dataset.csv',data_dir = '/srv/home/xsun256/Moleprop/summer19')

## leave out pubchem as test set
print("About to split")
splitter = wf.Splitter
indices,dataset = splitter.LOG(data, 'pubchem')

args = {'nb_epoch': 80,
        'batch_size': 50,
        'n_tasks': 1,
        'graph_conv_layers':[64,64],
        'dense_layer_size': 256,
#        'dropout': 0.0,
        'mode': 'regression'}


print("About to simulate")
rms,mae,prediction,test_dataset = wf.Exec.LOG_validation(dataset,indices, 'graphconv',model_args = args)

print("===============FINAL RESULTS==================")
print("RMSE = ", rms)
print("MAE = ", mae)
