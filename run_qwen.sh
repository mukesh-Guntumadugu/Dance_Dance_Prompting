#!/bin/bash
#SBATCH --job-name=qwen_onsets
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=06:00:00
#SBATCH --output=qwen_log_%j.txt

echo "=== Starting Qwen2-Audio Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

cd /data/mg546924/llm_beatmap_generator

# Block ~/.local (has old PyTorch 2.1.0 which breaks transformers>=4.45)
export PYTHONNOUSERSITE=1
# Inject new PyTorch from /data into Python path
export PYTHONPATH=/data/mg546924/torch_packages:$PYTHONPATH

# Use the conda env's Python directly (most reliable in SLURM batch scripts)
/home/mg546924/.conda/envs/qwenenv/bin/python extract_qwen_onsets.py

echo "=== Qwen Onset Detection Finished ==="
