#!/bin/bash
#SBATCH --job-name=breast-cancer
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --partition=long
#SBATCH --account=users-account
#SBATCH --mem=130GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user-email

### The example below is for breast cancer, where the user needs to concatenate all cell lines that belong to this tissue type. Users should repeat this process for each tissue type and molecular profile.


chemprop_train --data_path ../CNA/breast-cancer_train.csv --features_path ../CNA/breast-cancer_train_profile.csv --save_dir ../CNA/breast-cancer --no_features_scaling  --smiles_columns STD_SMILE --target_columns NLOGGI50_N --dataset_type regression --split_sizes 0.9 0.1 0.0 --no_features_scaling --epochs 30

### Prediction on breast cancer
chemprop_predict --test_path ../CNA/breast-cancer_test.csv --features_path ../CNA/breast-cancer_test_profile.csv --preds_path ../CNA/prediction/breast-cancer_preds.csv --no_features_scaling  --smiles_columns STD_SMILE --checkpoint_dir ../CNA/breast-cancer
