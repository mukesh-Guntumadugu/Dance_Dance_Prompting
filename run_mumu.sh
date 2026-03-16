#!/bin/bash
#SBATCH --job-name=mumu_onsets
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=06:00:00
#SBATCH --output=mumu_log_%j.txt

echo "=== Starting MuMu-LLaMA Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "==========================================="

cd /data/mg546924/llm_beatmap_generator

# Initialize conda so it works in a non-interactive SLURM job
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || \
eval "$(conda shell.bash hook)"

# Activate MuMu-LLaMA conda environment
conda activate mumullama

python3 extract_mumu_onsets.py

echo "=== MuMu-LLaMA Onset Detection Finished ==="
