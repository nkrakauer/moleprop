import workflow as wf

print("About to load")
loader = wf.Loader
data = loader.load(file_name = 'carroll_all.csv',data_dir = '/srv/home/xsun256/Moleprop/summer19/datasets')

print("About to split")
splitter = wf.Splitter
indices,dataset = splitter.k_fold(data, n_splits = 2)

args = None

dataset['flashpoint'] = dataset['flashPoint']

print("About to simulate")
scores,predictions,test_datasets = wf.Simulate.cv(dataset,indices, 'MPNN',model_args = None,n_splits = 2)

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
