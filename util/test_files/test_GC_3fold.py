import workflow as wf
import pandas as pd
import numpy as np

## Load dataset
print("About to load dataset\n")
data = wf.Loader.load('new_integrated_dataset.csv')
print("orginal dataset info:")
wf.Loader.getinfo(data)

## for removing pubchem 
# data = data[data.source != 'pubchem']

## split
print("about to split\n")
indices, newdata = wf.Splitter.k_fold(data,n_splits = 3,shuffle = True)

## Conduct CV
print("about to do CV\n")
scores,predictions,datasets = wf.Exec.cv(newdata, indices, 'GC', n_splits = 3)
rms,mae,rmss,maes = scores

print("==============FINAL RESULTS==================")
print("RMSE: ", rms)
print("MAE: ", mae)

# get plot for each cv iteration
for i in range(3):
    name = "GC_3fold_CV_"+str(i)+".png"
    print("CV ",i,": ")
    print("testset length: ",len(datasets[i]))
    print("pred length: ", len(predictions[i]))
    scores = {"RMSE":round(rmss[i],2), "MAE":round(maes[i],2)}
    wf.Plotter.parity_plot(predictions[i],datasets[i],plot_name = name, text = scores)
