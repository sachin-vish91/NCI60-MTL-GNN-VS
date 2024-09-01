#!/bin/bash
#SBATCH --job-name=train
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --partition=fast
#SBATCH --mem=150GB

# Activate conda env
conda activate chemprop


#CNA
chemprop_train --data_path ../CNA/CNA_train.csv --features_path ../CNA/CNA_train_profile.csv --smiles_columns STD_SMILE --target_columns NLOGGI50_N --dataset_type regression --save_dir ../CNA/check_point --split_sizes 0.9 0.1 0.0 --no_features_scaling --epochs 50 


#EXSNP
#chemprop_train --data_path ../EXSNP/EXSNP_train.csv --features_path ../EXSNP/EXSNP_train_profile.csv --smiles_columns STD_SMILE --target_columns NLOGGI50_N --dataset_type regression --save_dir ../EXSNP/check_point --split_sizes 0.9 0.1 0.0 --no_features_scaling --epochs 50 


#Methy
#chemprop_train --data_path ../Methy/Methy_train.csv --features_path ../Methy/Methy_train_profile.csv --smiles_columns STD_SMILE --target_columns NLOGGI50_N --dataset_type regression --save_dir ../Methy/check_point --split_sizes 0.9 0.1 0.0 --no_features_scaling --epochs 50 


#Protein
#chemprop_train --data_path ../Protein/Protein_train.csv --features_path ../Protein/Protein_train_profile.csv --smiles_columns STD_SMILE --target_columns NLOGGI50_N --dataset_type regression --save_dir ../Protein/check_point --split_sizes 0.9 0.1 0.0 --no_features_scaling --epochs 50 


#GeneExp
#chemprop_train --data_path ../GeneExp/GeneExp_train.csv --features_path ../GeneExp/GeneExp_train_profile.csv --smiles_columns STD_SMILE --target_columns NLOGGI50_N --dataset_type regression --save_dir ../GeneExp/check_point --split_sizes 0.9 0.1 0.0 --no_features_scaling --epochs 50 
