# autogluon 1.0
from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd
import random
import string
import numpy as np
import shutil

def remove_folder(path):
    try:
        # Use shutil.rmtree to remove the folder and its contents
        shutil.rmtree(path)
        print(f"Folder {path} and its contents removed successfully.")
    except OSError as e:
        print(f"Error: {e}")

folder_path = "./AutogluonModels"

validations_70 = {} #validation results on final 70% of dataset

for seed in range(0, 15):
  df = pd.read_csv('features.csv', names = [i for i in range(1281)])
  random.seed(seed)

  if seed not in validations_70:
        validations_70[seed] = {}

  # random selection of 70% aminoacid residues (rows) for training dataset
  dfl=pd.read_csv('features.csv',
           skiprows=lambda x: x > 0 and random.random() >=0.7, names= [i for i in range(1281)])
  dft = pd.concat([df, dfl]).drop_duplicates(keep=False)
  dft.to_csv("testr.csv", index=False)
  dfl.to_csv("studyr.csv", index=False)
  df.to_csv("allr.csv", index=False)

  train = TabularDataset('testr.csv')
  label = '1280'
  train_size = train.shape[0]
  train_data_size = int(train_size * 0.8)

  # dividing 30% of dataset into two parts: 20% and 10%
  train_data = train.sample(train_data_size, random_state=seed)
  test_set = train.drop(train_data.index)

  # randomly selecting different numbers of features for training on remainin 70% of dataset
  for f_number in [10,20,30,40,50,100,200,300, 500, 700, 1000,1250]:
      dfl=pd.read_csv('studyr.csv')
      random_columns = dfl.iloc[:, :1280].sample(f_number, axis=1)
      features = random_columns.columns.to_list()

      print (features)

      dfl_new = dfl[features+['1280']]
      dfl_new.to_csv("studyr_new.csv", index=False)
      train_data = TabularDataset('studyr_new.csv')
      predictor2 = TabularPredictor(label=label,eval_metric='f1').fit(train_data,presets=['best_quality'])
      p_val2 = predictor2.leaderboard().iloc[0]['score_val']
      validations_70[seed][f_number] = p_val2
      remove_folder(folder_path)

  with open('validation_random.txt', 'w') as my_file:
      print(validations_70,file=my_file)

