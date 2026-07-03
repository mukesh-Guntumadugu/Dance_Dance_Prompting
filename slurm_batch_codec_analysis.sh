#!/bin/bash
#SBATCH --job-name=codec_analysis
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/codec_analysis_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

echo "=== STARTING BATCH CODEC ANALYSIS (60-240 BPM) === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

echo "Running target script..."
# Using the qwenenv or .venv which has EnCodec and Librosa set up
/data/mg546924/conda_envs/qwenenv/bin/python -u batch_analyze_all_models.py

echo "=== DONE $(date) ==="
