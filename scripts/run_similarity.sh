#!/bin/bash
#SBATCH --job-name=Similarity
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --partition=long
#SBATCH --account=Name of the account
#SBATCH --mem=100GB
#SBATCH --mail-type=ALL
#SBATCH --mail-user=user_name@gmail.com


python similarity.py
