#!/bin/bash
#SBATCH --job-name=train_mumu_bpm
#SBATCH --output=logs/train_mumu_bpm_%j.log
#SBATCH --error=logs/train_mumu_bpm_%j.log
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL

set -e
cd /data/mg546924/llm_beatmap_generator

echo "=============================================="
echo "Training MuMu-LLaMA (5 Epochs | BPM Dataset)"
echo "Job ID      : $SLURM_JOB_ID"
echo "Start       : $(date)"
echo "=============================================="

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0

/data/mg546924/conda_envs/qwenenv/bin/python scripts/train_mumu_bpm_lora.py

echo "✅ MuMu BPM Training Complete: $(date)"
