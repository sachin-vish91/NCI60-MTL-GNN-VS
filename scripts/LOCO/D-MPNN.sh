#!/bin/bash
#SBATCH --job-name=CNA
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --partition=long
#SBATCH --account=users-account
#SBATCH --mem=130GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user-email

### Activate conda env
conda activate chemprop

### The example below is for the CNA profile and MCF7 cell line. Users should repeat this process for all cell lines and each molecular profile..

chemprop_train --data_path ../CNA/MCF7_train.csv --features_path ../CNA/MCF7_train_profile.csv --save_dir ../CNA/MCF7 --no_features_scaling  --smiles_columns STD_SMILE --target_columns NLOGGI50_N --dataset_type regression --split_sizes 0.9 0.1 0.0 --no_features_scaling --epochs 30

### Prediction on MCF7 cell line.
chemprop_predict --test_path ../CNA/MCF7_test.csv --features_path ../CNA/MCF7_test_profile.csv --preds_path ../CNA/prediction/MCF7_preds.csv --no_features_scaling  --smiles_columns STD_SMILE --checkpoint_dir ../CNA/MCF7
