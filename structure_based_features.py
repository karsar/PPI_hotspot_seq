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

validations_70 = {}

"""
    energetic.csv contains precalculated structure and sequence based features, 
    such as gas energy, conservation score and etc.
"""

for seed in range(0, 50):
  df = pd.read_csv('energetic.csv', low_memory=False)
  random.seed(seed)


  # random selection of 70% aminoacid residues (rows) for training dataset
  dfl=pd.read_csv('/content/drive/MyDrive/energetic.csv', low_memory=False,
           skiprows=lambda x: x > 0 and random.random() >=0.7)
  dft = pd.concat([df, dfl]).drop_duplicates(keep=False)
  dft.to_csv("testb.csv", index=False)
  dfl.to_csv("studyb.csv", index=False)
  df.to_csv("allb.csv", index=False)

  train = TabularDataset('studyb.csv')
  label = 'labels'
  predictor2 = TabularPredictor(label=label,eval_metric='f1').fit(train,presets=['best_quality'])
  p_val2 = predictor2.leaderboard().iloc[0]['score_val']
  validations_70[seed] = p_val2
  remove_folder(folder_path)

  with open('validation_structure.txt', 'w') as my_file:
      print(validations_70,file=my_file)

