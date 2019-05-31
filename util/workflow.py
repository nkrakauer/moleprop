import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

# pkg needed for DeepChem
import tensorflow as tf
import deepchem as dc
# from deepchem.models.tensorgraph.models.graph_models import GraphConvModel


class Loader:
    data = pd.DataFrame()
    def load(file_name, data_dir = './'):
        """
        load data from .csv file
        """
        # TODO: need to clean data before loading?
        data_file = os.path.join(data_dir, file_name)
        if not os.path.exists(data_file):
            if data_dir == './':
                data_dir = "current"
            error_msg = file_name + " was not found in " + data_dir + " directory"
            sys.exit(error_msg)
        print("|||||||||||||||||||||Loading " + file_name+ "|||||||||||||||||||||||")
        data = pd.read_csv(data_file) # encoding='latin-1' might be needed
        return data

    def getinfo():
        """
        get information of the dataset
        """
        sources = data.source.unique()
        source_info = dict()
        for s in sources:
            counter = 0
            for i in range(0,len(data.index)):
                if data.iloc[i]['source'] == s:
                    counter += 1
            source_info[s] = [counter]
        print('-----------------------------------------------------')
        print("Dataset length is: ", len(data.index))
        print('-----------------------------------------------------')
        print("Dataset sources info: ")
        for s in sources:
            print("  Source Name: " + s + ", Number of Data: " + str(source_info[s]))
        print('-----------------------------------------------------')



class Splitter:

    def k_fold(dataset, n_splits = 3, shuffle = True):
        """
        split data into k-fold
        return indices of training and test sets
        """
        if shuffle == True:
            random_state = 4396
        kf = KFold(n_splits, shuffle, random_state)
        indices = kf.split(dataset)
        return indices

    def LOG(dataset, test_group):  # leave out group
        """
        split dataset by leaving out a specific source as test set
        """
        test_indices = []
        train_indices = list(range(len(dataset.index)))
        print("||||||||||||||||||| "+test_group+ " will be used as test set|||||||||||||||||||")
        for i in range(0,len(dataset.index)):
            if dataset.iloc[i]['source'] == test_group:
                test_indices.append(i)
                train_indices.remove(i)
        return (train_indices, test_indices)
    
    def leave_out_moleClass(dataset, mole_class_to_leave_out):
        """
        TODO:
        leave out a specific mole class
        """
        return 0



class Simulate:
        
    def cv(data,
           indices, # k-fold indices of training and test sets
           model,  # need to be either MPNN or GraphConv
           model_args = None, 
           n_splits = 3):   
        """
        pass data into models (MPNN or graphconv) and conduct cross validation
        """
        if not (model == 'MPNN' or model == 'graphconv'):
            sys.exit("Only support MPNN model and GraphConv model")
        cv_scores = []
        for train_indices, test_indices in indices:
            train_set = data.iloc[train_indices]
            test_set = data.iloc[test_indices]
            train_set.to_csv('train_set.csv',index = False)
            test_set.to_csv('test_set.csv',index = False)
            if model == 'MPNN':
                score = Model.MPNN(model_args, "train_set.csv", "test_set.csv")
            elif model == 'graphconv':
                score = Model.graphconv(model_args,"train_set.csv", "test_set.csv")       
            cv_scores.append(score)
        avg_cv_score = sum(cv_scores)/n_splits
        return avg_cv_score

    def LOG_validation(data,
                       indices, 
                       model, 
                       model_args = None):
        """
        Conduct leave-out-group validation
        """
        
        if not (model == 'MPNN' or model == 'graphconv'):
            sys.exit("Only supports MPNN model and graphconv model")
        train_indices, test_indices = indices
        train_set = data.iloc[train_indices]
        test_set = data.iloc[test_indices]
        train_set.to_csv('train_set.csv',index = False)
        test_set.to_csv('test_set.csv',index = False)
        if model == 'MPNN':
            score = Model.MPNN(model_args, "train_set.csv", "test_set.csv")
        elif model == 'GraphConv':
            score = Model.graphconv(model_args, "train_set.csv", "test_set.csv")       
        return score





class Model:
    
    default_args = {
        'graphconv': {
            'nb_epoch': 50, 
            'batch_size': 64, 
            'n_tasks': 1, 
            'graph_conv_layers':[64,64],
            'dense_layer_size': 256,
            'dropout': 0,
            'mode': 'regression'},
        'MPNN':{
            'n_tasks':1,
            'n_atom_feat':75,
            'n_pair_feat':14,
            'T':1,
            'M':1,
            'batch_size':32,
            'nb_epoch': 50,
            'learning_rate':0.0001,
            'use_queue':False,
            'mode':"regression",
            'n_hidden' :75
        }
    }
    
    def graphconv(args, train_set, test_set):
        # parse arguments
        if args == None:
            args = Model.default_args['graphconv']
        nb_epoch = args["nb_epoch"]
        batch_size =  args["batch_size"]
        n_tasks = args["n_tasks"]
        graph_conv_layers = args["graph_conv_layers"]
        dense_layer_size = args["dense_layer_size"]
        dropout = args["dropout"]
        mode = args["mode"]  # regression or classificiation

        flashpoint_tasks = ['flashPoint']  # Need to set the column name to be excatly "flashPoint"
        loader = dc.data.CSVLoader(tasks = flashpoint_tasks, 
                                        smiles_field="smiles", 
                                        featurizer = dc.feat.ConvMolFeaturizer())
        train_dataset = loader.featurize(train_set, shard_size=8192)
        test_dataset = loader.featurize(test_set, shard_size=8192)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
            train_dataset = transformer.transform(train_dataset)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=test_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
             test_dataset = transformer.transform(test_dataset)
                
                
        model = dc.models.GraphConvModel(n_tasks = n_tasks, mode = mode, dropout = dropout)
        metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
        model.fit(train_dataset, batch_size = batch_size, nb_epoch = nb_epoch) 
        score = list( model.evaluate(test_dataset, [metric],transformers).values()).pop()
        print("=================================")
        print("GraphConv\n -----------------------------\n RMSE score is: ", score)        
        print("=================================")        return score

    def MPNN(args, train_set, test_set):
        # parse arguments
        if args == None:
            args = Model.default_args['MPNN']       
        n_tasks = args['n_tasks']
        n_atom_feat = args['n_atom_feat']
        n_pair_feat = args['n_pair_feat']
        T = args['T']
        M = args['M']
        batch_size = args['batch_size']
        learning_rate = args['learning_rate']
        use_queue = args['use_queue']
        mode = args['mode']
        nb_epoch = args['nb_epoch']
        n_hidden = args['n_hidden]

        flashpoint_tasks = ['flashPoint']  # Need to set the column name to be excatly "flashPoint"
        loader = dc.data.CSVLoader(tasks = flashpoint_tasks, 
                                        smiles_field="smiles", 
                                        featurizer = dc.feat.WeaveFeaturizer())

        train_dataset = loader.featurize(train_set, shard_size=8192)
        test_dataset = loader.featurize(test_set, shard_size=8192)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=train_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
            train_dataset = transformer.transform(train_dataset)
        transformers = [
            dc.trans.NormalizationTransformer(
            transform_y=True, dataset=test_dataset, move_mean=True) # sxy: move_mean may need to change (3/23/2019)
        ]
        for transformer in transformers:
             test_dataset = transformer.transform(test_dataset)

        model = dc.models.MPNNModel(n_tasks = n_tasks,
                                    n_atom_feat=n_atom_feat,
                                    n_pair_feat=n_pair_feat,
                                    n_hidden= n_hidden,
				                    T=T,
                                    M=M,
                                    batch_size=batch_size,
                                    learning_rate=learning_rate,
                                    use_queue=True,#use_queue,
                                    mode=mode)        
                
        metric = dc.metrics.Metric(dc.metrics.rms_score, np.mean)
        model.fit(train_dataset, nb_epoch = nb_epoch) 
        score = list( model.evaluate(test_dataset, [metric],transformers).values()).pop()
        print("=================================")
        print("MPNN\n -----------------------------\n RMSE score is: ", score)        
        print("=================================")
        return score