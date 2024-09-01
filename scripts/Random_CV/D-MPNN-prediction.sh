#!/bin/bash
#SBATCH --job-name=prediction
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --partition=fast
#SBATCH --mem=150GB

# Activate conda env
conda activate chemprop

chemprop_predict --test_path test.csv --checkpoint_path ../model.pt --preds_path prediction.csv
