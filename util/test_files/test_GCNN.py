import workflow as wf
import pandas as pd

print("About to load")
loader = wf.Loader
data = loader.load(file_name = 'carroll_all.csv',data_dir = '/srv/home/xsun256/Moleprop/summer19/datasets')

print("About to split")
splitter = wf.Splitter
indices,dataset = splitter.k_fold(data, n_splits = 2)

args = {'nb_epoch': 80,
        'batch_size': 50,
        'n_tasks': 1,
        'graph_conv_layers':[64,64],
        'dense_layer_size': 256,
#        'dropout': 0.0,           # for testing if this workflow tool can correctly use default dropout if it is not inputted
        'mode': 'regression'}

dataset['flashpoint'] = dataset['flashPoint']

print("About to simulate")
scores,predictions,test_datasets = wf.Simulate.cv(dataset,indices, 'graphconv',model_args = args,n_splits = 2)

# save predictions and corresponding testsets for testing parity_plot function
#print(test_datasets[1])
t = pd.DataFrame(data = test_datasets[1])
t.to_csv("./test_dataset.csv")

f = open("./pred.txt",'w')
predc = predictions[1].tolist()
pred = " ".join(str(x) for x in predc)
f.write(pred)
f.close()

print("=========================Final Result======================")

rms,mae,rms_all,mae_all = scores
print("rms score is : ", rms)
print("mae score is : ", mae)

print("rms_all is: ")
print(rms_all)

print("mae_all is: ")
print(mae_all)

for i in range(2):
    print("predictions[",i,"]:")
    print(predictions[i])
    
