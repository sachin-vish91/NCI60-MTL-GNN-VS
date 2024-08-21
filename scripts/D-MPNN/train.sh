#!/bin/bash
#SBATCH --job-name=train-HCT-116
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --partition=fast
#SBATCH --mem=150GB

# Activate conda env
conda activate chemprop

chemprop_train --data_path Htrain.csv --config_path  Config.json --smiles_columns SMILE --target_columns NLOGGI50 --dataset_type regression --epochs 100 --save_dir ./output_dir --split_sizes 0.9 0.1 0.0
