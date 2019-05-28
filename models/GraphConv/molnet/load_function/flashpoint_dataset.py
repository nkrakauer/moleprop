"""
Loader for flashpoint dataset collected by Nathaniel.
Author: Sean Sun
2019/2/16
"""
from __future__ import division
from __future__ import unicode_literals

import os
import logging
import deepchem

logger = logging.getLogger(__name__)


def load_flashpoint(usr_data_dir, featurizer='ECFP', split='index', reload=True, move_mean=True):
  """Load flashpoint datasets."""
  logger.info("About to featurize flashpoint dataset.")
  logger.info("About to load flashpoint dataset.")

  data_dir = usr_data_dir  #Hardcoded dir_address, need to be modified
  if reload:
    if move_mean:
      dir_name = "flahspoint/" + featurizer + "/" + str(split)
    else:
      dir_name = "falshpoint/" + featurizer + "_mean_unmoved/" + str(split)
    save_dir = os.path.join(data_dir, dir_name)

  dataset_file = os.path.join(data_dir, "data.csv")
  if not os.path.exists(dataset_file):
      print("data.csv was not found")

  flashpoint_tasks = ['flashPoint']

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return flashpoint_tasks, all_dataset, transformers

  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()

  loader = deepchem.data.CSVLoader(
      tasks=flashpoint_tasks, smiles_field="smiles", featurizer=featurizer)
  dataset = loader.featurize(dataset_file, shard_size=8192)

  # Initialize transformers
  transformers = [
      deepchem.trans.NormalizationTransformer(
          transform_y=True, dataset=dataset, move_mean=move_mean)
  ]

  logger.info("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  if split == None:
    return flashpoint_tasks, (dataset, None, None), transformers

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  train, valid, test = splitter.train_valid_test_split(dataset)
  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)
  return flashpoint_tasks, (train, valid, test), transformers
