#!/bin/bash
#SBATCH --job-name=general_test
#SBATCH --output=generalization_test_%j.out
#SBATCH --error=generalization_test_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=defq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

cd $SLURM_SUBMIT_DIR

# Activate environment with transformers and torch
source /data/mg546924/miniconda3/bin/activate /home/mg546924/.conda/envs/mumullama

echo "Running Generalization Test & BPM Predictor..."
python3 -u test_generalization.py

echo "Done! Check Pred_vs_Target_BPM.png for the predictions!"
