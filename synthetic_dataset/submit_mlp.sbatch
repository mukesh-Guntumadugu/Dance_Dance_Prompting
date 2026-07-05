#!/bin/bash
#SBATCH --job-name=train_mlp
#SBATCH --output=train_mlp_%j.out
#SBATCH --error=train_mlp_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=defq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

cd $SLURM_SUBMIT_DIR

# Activate environment with transformers, torch, and scikit-learn
source /data/mg546924/miniconda3/bin/activate /home/mg546924/.conda/envs/mumullama

echo "Starting PyTorch MLP training..."
python3 -u train_mlp.py

echo "Training complete! Check MLP_Pred_vs_Target_BPM.png"
