# PPI_hotspot_seq

This repository contains the supporting code and data for the research paper titled "Applying Protein Language Models Using Limited Dataset: Sequence-Based Hot Spot Prediction in Protein Interactions Using AutoGluon."

To predict hotspots using your own sequences try [Open in Google Colab](https://colab.research.google.com/github/karsar/PPI_hotspot_seq/blob/main/ESM2_predict_PPI_hotspots.ipynb)

## Contents

- **dataset_entries.csv:** Contains entries of PPI-Hotspot+PDB(BM(1.1)) dataset.
- **features.csv:** Contains prepared ESM-2 embeddings of amino acid residues from the PPI-Hotspot+PDB(BM(1.1)) dataset.
- **energetic.csv:** Contains sequence and structure-based features for amino acid residues from the PPI-Hotspot+PDB(BM(1.1)) dataset.
- **importance_based_features.py:** Code for experimenting with importance-based features.
- **randomly_selected_features.py:** Code for experimenting with randomly selected features.
- **structure_based_features.py:** Code for obtaining predictions using a conventional sequence/structure-based approach.

