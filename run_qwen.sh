#!/bin/bash
#SBATCH --job-name=qwen_onsets
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=06:00:00
#SBATCH --output=qwen_log_%j.txt

echo "=== Starting Qwen2-Audio Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================="

# Force Python to ignore ~/.local site-packages (prevents shadowing conda env)
export PYTHONNOUSERSITE=1
# Force pip to install to datapool instead of ~/.local
export PIP_USER_BASE=/datapool001/data/mg546924/.local
export PATH=$PIP_USER_BASE/bin:$PATH
export PYTHONPATH=$PIP_USER_BASE/lib/python3.9/site-packages:$PYTHONPATH

# Use the conda env's Python directly (most reliable in SLURM batch scripts)
/home/mg546924/.conda/envs/qwenenv/bin/python extract_qwen_onsets.py

echo "=== Qwen Onset Detection Finished ==="
