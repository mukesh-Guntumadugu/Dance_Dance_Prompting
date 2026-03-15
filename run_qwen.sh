#!/bin/bash
#SBATCH --job-name=qwen_onsets
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=06:00:00
#SBATCH --output=qwen_log_%j.txt

echo "=== Starting Qwen2-Audio Onset Detection ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "============================================="

cd /data/mg546924/llm_beatmap_generator

python3 extract_qwen_onsets.py

echo "=== Qwen Onset Detection Finished ==="
