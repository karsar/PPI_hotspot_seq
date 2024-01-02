# autogluon 1.0
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import random
import string
import numpy as np
import shutil

# to save space, not needed models are being deleted
def remove_folder(path):
    try:
        # Use shutil.rmtree to remove the folder and its contents
        shutil.rmtree(path)
        print(f"Folder {path} and its contents removed successfully.")
    except OSError as e:
        print(f"Error: {e}")

folder_path = "./AutogluonModels"

validations_20 = {} #validation results on initial 20% of dataset
validations_70 = {} #validation results on final 70% of dataset

""" features.csv file contains prepared embeddings, which are 
    obtained as:
    import torch
    import esm (large language model for proteins)
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    data = [('some_id;', "100_letter_aminoacid_sequence")]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    with torch.no_grad():
         results = model(batch_tokens, repr_layers=[33], return_contacts=True)
         x = (results['representations'][33][0,49,:]) # for a 1280 length vector for an amino acid in the middle of the sequence   
"""
  
for seed in range(0, 15):
  df = pd.read_csv('features.csv', names = [i for i in range(1281)])
  random.seed(seed)

  if seed not in validations_70:
        validations_70[seed] = {}

  # random selection of 70% aminoacid residues (rows) for training dataset
  dfl=pd.read_csv('features.csv',
           skiprows=lambda x: x > 0 and random.random() >=0.7, names= [i for i in range(1281)])
  dft = pd.concat([df, dfl]).drop_duplicates(keep=False)
  dft.to_csv("test.csv", index=False)
  dfl.to_csv("study.csv", index=False)
  df.to_csv("all.csv", index=False)

  train = TabularDataset('test.csv')
  label = '1280'
  train_size = train.shape[0]
  train_data_size = int(train_size * 0.8)
   
  # dividing 30% of dataset into two parts: 20% and 10%
  train_data = train.sample(train_data_size, random_state=seed)
  test_set = train.drop(train_data.index)

  predictor = TabularPredictor(label=label,eval_metric='f1').fit(train_data,presets=['best_quality'])
  p_val = predictor.leaderboard().iloc[0]['score_val']
  validations_20[seed] = p_val

  d = predictor.feature_importance(test_set, num_shuffle_sets = 20)
  d = d.sort_values(by='importance', ascending=False)
  d.to_csv('imp_'+str(seed)+'.csv')
  remove_folder(folder_path)

  # selecting different numbers of features for training on remainin 70% of dataset
  for f_number in [10,20,30,40,50,100,200,300, 500, 700,1000,1280]: # f_number = k number of features selected
      x = d.head(f_number)
      print (x.head())
      features = x.index.values.tolist()+['1280']
      print (features)
      dfl=pd.read_csv('study.csv')
      dfl_new = dfl[features]
      dfl_new.to_csv("study_new.csv", index=False)
      train_data = TabularDataset('study_new.csv')
      predictor2 = TabularPredictor(label=label,eval_metric='f1').fit(train_data,presets=['best_quality'])
      p_val2 = predictor2.leaderboard().iloc[0]['score_val']
      validations_70[seed][f_number] = p_val2
      remove_folder(folder_path)

  with open('validations.txt', 'w') as my_file:
      print(validations_20, file=my_file)
      print(validations_70,file=my_file)
