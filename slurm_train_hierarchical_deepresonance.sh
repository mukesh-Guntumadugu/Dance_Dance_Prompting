#!/bin/bash
#SBATCH --job-name=train_hierarchical_deepresonance
#SBATCH --output=logs/train_hierarchical_deepres_%j.log
#SBATCH --error=logs/train_hierarchical_deepres_%j.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL

set -e
cd /data/mg546924/llm_beatmap_generator

echo "=============================================="
echo "Training DeepResonance Hierarchical Director"
echo "Job ID      : $SLURM_JOB_ID"
echo "Start       : $(date)"
echo "=============================================="

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))
export DS_SKIP_CUDA_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [ ! -f "scripts/cluster_to_patterns_tokens.txt" ]; then
    echo "[ERROR] ERROR: cluster_to_patterns_tokens.txt not found!"
    exit 1
fi

/data/mg546924/conda_envs/qwenenv/bin/python scripts/train_hierarchical_deepresonance.py

echo "✅ DeepResonance Hierarchical Training Complete: $(date)"
