import workflow as wf

# load dataset
loader = wf.Loader
data = loader.load('carroll_all.csv')

# split dataset into k-fold or LOG
splitter = wf.Splitter
indices = splitter.k_fold(data, n_splits = 5)

# set model_arguments
'''
GraphConv_args = {'nb_epoch': 80,
        'batch_size': 50,
        'n_tasks': 1,
        'graph_conv_layers':[64,64],
        'dense_layer_size': 256,
        'dropout': 0.0,
        'mode': 'regression'}
'''

# train and test the model
score = wf.Simulate.cv(data,indices, 'MPNN',model_args = None,n_splits = 5)

print("\n===============FINAL avg CV score================")
print(score)
