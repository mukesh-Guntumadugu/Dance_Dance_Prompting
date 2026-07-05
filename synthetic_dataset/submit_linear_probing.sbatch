#!/bin/bash
#SBATCH --job-name=lin_prob_bpm
#SBATCH --output=linear_probing_%j.out
#SBATCH --error=linear_probing_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodelist=node007
#SBATCH --partition=defq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

cd $SLURM_SUBMIT_DIR

# Activate environment with transformers and torch
source /data/mg546924/miniconda3/bin/activate /home/mg546924/.conda/envs/mumullama

echo "Running linear probing on Whisper embeddings..."
python3 -u run_linear_probing.py

echo "Done! Check linear_probing_correlations.csv and linear_probing_top_dims.png"
