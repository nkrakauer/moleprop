import sys
sys.path.append('../../../util')
import workflow as wf
import pandas as pd
from operator import itemgetter
from deepchem.models.tensorgraph.models.graph_models import GraphConvModel, MPNNModel

data = wf.Loader.load(data_dir='../../../data',file_name = 'didier12.csv')

train = [i for i in range(166)]
test = [i for i in range(166,210)]
ind = [(train,test)]

params_dict = {
        "nb_epoch":[70,100,150,200,400],
        "n_tasks":[1],
        "batch_size": [8,32],
        "graph_conv_layers":[[64,64]],
        "dense_layer_size":[128,256,512],
        "learning_rate":[0.005,0.0005],
        "mode":['regression']
}
mpnn_dict = {
      "nb_epoch":[70,100,150,200,400],
      "n_tasks":[1],
      "batch_size": [8,32],
      "n_atom_feat": [75],
      "n_pair_feat": [14],
      "T": [1],
      "M": [1],
     "dropout": [0.0, 0.2, 0.4],
      "learning_rate":[0.005,0.0005, 0.001],
      "mode":['regression']
     }

def mpnn_model_builder(model_params , model_dir):
    return MPNNModel(**model_params, model_dir = "./models")

def gc_model_builder(model_params , model_dir):
    gc_model = GraphConvModel(**model_params, model_dir = "./models")
    return gc_model
i = 0
for train,test in ind:
    train_set = data.iloc[train]
    test_set = data.iloc[test]
    train_set.to_csv('train_'+str(i)+'.csv')
    test_set.to_csv('test_'+str(i)+'.csv')
    optimizer = wf.HyperparamOpt(mpnn_model_builder)
    best_model, best_hyperparams, all_results = optimizer.CVgridsearch(mpnn_dict,train_set)
    file = open('opt_result_'+str(i)+'.txt', 'w')
    s = 'Best Hyperparameter:' + str(best_hyperparams) + '\n\nAll Results:\n'
    file.write(s)
    from operator import itemgetter
    for k, v in sorted(all_results.items(), key=itemgetter(1)):
        s =  k + " : " + str(v)+"\n"
        file.write(s)
    file.close()
