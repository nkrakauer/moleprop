import os
import sys
import numpy as np
import pandas as pd
import random                                            # for randomly choosing indices from left-out group as test indices
import bokeh                                             # for interactive plot
import statistics as stat
import seaborn
import matplotlib.pyplot as plt
import integration_helpers                               # for removing duplicates
from sklearn.model_selection import KFold
from deepchem.trans.transformers import undo_transforms  # for getting real predictions

## pkg needed for DeepChem
import tensorflow as tf
import deepchem as dc
# from deepchem.models.tensorgraph.models.graph_models import GraphConvModel


class Loader:
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

    def getinfo(data):
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
        print("Mean: " + str(data['flashpoint'].mean()))
        print('-----------------------------------------------------')
        print("Std: " + str(data['flashpoint'].std()))



class Splitter:
	
    def k_fold(dataset, n_splits = 3, shuffle = True, random_state = None):
        """
        split data into k-fold

        Return:
        indices of k-fold training and test sets
        new dataset after removing duplicates
        """
        dataset = integration_helpers.remove_duplicates(dataset) # remove duplicates
        if shuffle == True:
            random_state = 4396
        kf = KFold(n_splits, shuffle, random_state)
        indices = kf.split(dataset)
        return indices, dataset

    def LOG(dataset, test_group, frac = 1):  # leave out group
        """
        split dataset by leaving out a specific source as test set
        
        params:
        dataset: data frame
        test_group: string
        frac: fraction of the left-out group that will be used as test set
        """
        # remove duplicates in train group.
        test_df = dataset[dataset['source'] == test_group]
        train_df = dataset[dataset['source'] != test_group]

        # remove data points in  train dataframe that match smiles strings in
        # test dataframe
        for index, row in test_df.iterrows():
            smi = row['smiles']
            train_df = train_df[train_df['smiles'] != smi]
        frames = [train_df, test_df]
        dataset = pd.concat(frames)
        dataset.reset_index(drop=True, inplace=True)
        raw_test_indices = []
        raw_train_indices = list(range(len(dataset.index)))
        print("||||||||||||||||||| "+test_group+ " will be used as test set|||||||||||||||||||")
        for i in range(0,len(dataset.index)):
            if dataset.iloc[i]['source'] == test_group:
                raw_test_indices.append(i)
                raw_train_indices.remove(i)
        test_indices = random.sample(raw_test_indices, int(frac*len(raw_test_indices)))
        # print(test_indices)
        raw_test_indices = [x for x in raw_test_indices if x not in test_indices]
        train_indices = raw_train_indices + raw_test_indices
        return (train_indices, test_indices), dataset
    
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
        
        Return:
        avg_cv_rms_score
        avg_cv_mae_score
        cv_rms_scores: a list of RMSE scores from cross validation
        cv_mae_scores: a list of MAE scores from cross validation
        """
        if not (model == 'MPNN' or model == 'graphconv' or model == 'GC' or model == 'GraphConv'):
            sys.exit("Only support MPNN model and GraphConv model")
        cv_rms_scores = []
        cv_mae_scores = []
        cv_predictions = []
        cv_test_datasets = []
        for train_indices, test_indices in indices:
            train_set = data.iloc[train_indices]
            test_set = data.iloc[test_indices]
            cv_test_datasets.append(test_set)        
            train_set.to_csv('train_set.csv',index = False)
            test_set.to_csv('test_set.csv',index = False)
            if model == 'MPNN':
                rms_score,mae_score,pred = Model.MPNN(model_args, "train_set.csv", "test_set.csv")
            elif model == 'graphconv' or model == 'GC' or model == 'GraphConv':
                rms_score,mae_score,pred = Model.graphconv(model_args,"train_set.csv", "test_set.csv")       
            cv_rms_scores.append(rms_score)
            cv_mae_scores.append(mae_score)
            cv_predictions.append(pred)
            os.remove("train_set.csv")
            os.remove("test_set.csv")
        avg_cv_rms_score = sum(cv_rms_scores)/n_splits
        avg_cv_mae_score = sum(cv_mae_scores)/n_splits
        scores = (avg_cv_rms_score,avg_cv_mae_score,cv_rms_scores,cv_mae_scores)
        return scores, cv_predictions, cv_test_datasets

    def LOG_validation(data,
                       indices, 
                       model, 
                       model_args = None):
        """
        Conduct leave-out-group validation
        """
        
        if not (model == 'MPNN' or model == 'graphconv' or model == 'GC' or model == 'GraphConv'):
            sys.exit("Only supports MPNN model and graphconv model")
        train_indices, test_indices = indices
        train_set = data.iloc[train_indices]
        test_set = data.iloc[test_indices]
        train_set.to_csv('train_set.csv',index = False)
        test_set.to_csv('test_set.csv',index = False)
        if model == 'MPNN':
            rms_score,mae_score,pred = Model.MPNN(model_args, "train_set.csv", "test_set.csv")
        elif model == 'GraphConv' or model == 'graphconv' or model == 'GC':
            rms_score,mae_score,pred = Model.graphconv(model_args,"train_set.csv", "test_set.csv")       
        os.remove("train_set.csv")
        os.remove("test_set.csv")
        return rms_score,mae_score,pred,test_set

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
            'n_pair_feat':14,      # NEED to be 14 for WaveFeaturizer
            'T':1,
            'M':1,
            'batch_size':32,
            'nb_epoch': 50,
            'learning_rate':0.0001,
            'use_queue':False,
            'mode':"regression"
        }
    }
    
    def graphconv(args, train_set, test_set):
        # parse arguments
        model_args = Model.default_args['graphconv']
        if args != None:
            for key in args:
                model_args[key] = args[key]
                print(key + " is: " + str(args[key]))
        flashpoint_tasks = ['flashpoint']  # Need to set the column name to be excatly "flashPoint"
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
                
                
        model = dc.models.GraphConvModel(n_tasks = model_args['n_tasks'], 
                                         mode = model_args['mode'], 
                                         dropout = model_args['dropout'])
        metric_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean) # RMSE score
        metric_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean) # MAE score
        model.fit(train_dataset, nb_epoch = model_args['nb_epoch']) 
        pred = model.predict(test_dataset)
        pred = undo_transforms(pred, transformers)
        rms_score = list( model.evaluate(test_dataset, [metric_rms],transformers).values()).pop()
        mae_score = list( model.evaluate(test_dataset, [metric_mae],transformers).values()).pop()
        return rms_score, mae_score,pred

    def MPNN(args, train_set, test_set):
        # parse arguments
        model_args = Model.default_args['MPNN']
        if args != None:
            for key in args:
                model_args[key] = args[key]

        flashpoint_tasks = ['flashpoint']  # Need to set the column name to be excatly "flashPoint"
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
        model = dc.models.MPNNModel(n_tasks = model_args['n_tasks'],
                                    n_atom_feat = model_args['n_atom_feat'],
                                    n_pair_feat = model_args['n_pair_feat'],
                                    T = model_args['T'],
                                    M = model_args['M'],
                                    batch_size = model_args['batch_size'],
                                    learning_rate = model_args['learning_rate'],
                                    use_queue = model_args['use_queue'],
                                    mode = model_args['mode'])                    
        metric_rms = dc.metrics.Metric(dc.metrics.rms_score, np.mean) # RMSE score
        metric_mae = dc.metrics.Metric(dc.metrics.mae_score, np.mean) # MAE score
        model.fit(train_dataset, nb_epoch = model_args['nb_epoch'])
        pred = model.predict(test_dataset)
        pred = undo_transforms(pred, transformers)
        rms_score = list( model.evaluate(test_dataset, [metric_rms],transformers).values()).pop()
        mae_score = list( model.evaluate(test_dataset, [metric_mae],transformers).values()).pop()
        return rms_score,mae_score,pred

			
class Plotter:
    
    def parity_plot(pred, test_dataset, errorbar = False):
        """
        pred_rsutl: List - predicted results
        test_dataset: DataFrame - original test dataset containing index, true flashpoints, source
        errorbar: if true, plot scatter plot for error bars
        """
        # add pred_result to the test_dataset DataFrame
        seaborn.set(style = 'white',font_scale = 2)
        yeer = []
        avg_pred = []
        if errorbar == True:
            for i in range(len(test_dataset)):
                yeer.append(max(pred[i]) - min(pred[i]))
                avg_pred.append(stat.mean(pred[i]))
        else:
            avg_pred = pred
            yeer = [0]*len(test_dataset)
        test_dataset['pred'] = avg_pred
        test_dataset['yeer'] = yeer
        x = list(test_dataset['flashpoint'].values)
        y = list(test_dataset['pred'].values)
        # set figure parameters
        fg = seaborn.FacetGrid(data=test_dataset, hue='source', height = 8, aspect=1)
        fg.map(plt.errorbar,                  # type of plot
               'flashpoint', 'pred', 'yeer',  # data column
               fmt = 'o', markersize = 5     # args for errorbar
              ).add_legend()                  # add legend
        # set x,y limit
        min_val = min(min(y),min(y)-max(yeer),min(x)-max(yeer))
        max_val = max(max(y),max(y)+max(yeer),max(x)+max(yeer))
        fg.set(xlim = (min_val,max_val), ylim =(min_val, max_val))
        for ax in fg.axes.flat:
            ax.plot((min_val, max_val),(min_val, max_val))
        plt.title("Parity Plot") 
        plt.ylabel("Predicted") 
        plt.xlabel("Experimental") 
        seaborn.despine(fg.fig,top=False, right=False)#, left=True, bottom=True,)
	plt.savefig('parity_plot.png')
    
    def interactive_plot(pred_result,true_result):
        return 0
