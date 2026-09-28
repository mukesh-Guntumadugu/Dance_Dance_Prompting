#!/bin/bash
#SBATCH --job-name=deep_hi_fix
#SBATCH --output=logs/train_hierarchical_deepres_%j.log
#SBATCH --error=logs/train_hierarchical_deepres_%j.log
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --exclude=node002
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL

# 1-epoch validation run of the fixed training script (fix v3)
# Changes vs 49125: removed expandable_segments (not supported by this PyTorch),
# fixed vicuna path to use the same one the model itself uses (7b_v0),
# and model.llama_tokenizer is already set by DeepResonanceModel.__init__.

set -e
cd /data/mg546924/llm_beatmap_generator

echo "=============================================="
echo " DeepResonance Hierarchical Director - FIX v3"
echo " Job ID : $SLURM_JOB_ID"
echo " Start  : $(date)"
echo " Node   : $(hostname)"
echo "=============================================="

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export CUDA_HOME=$(dirname $(dirname $(which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)))
export DS_SKIP_CUDA_CHECK=1
# NOTE: expandable_segments is NOT supported by the older PyTorch in deepresonance_env
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

if [ ! -f "scripts/cluster_to_patterns_tokens.txt" ]; then
    echo "[ERROR] cluster_to_patterns_tokens.txt not found!"
    exit 1
fi

/data/mg546924/conda_envs/deepresonance_env/bin/python \
    scripts/train_hierarchical_deepresonance_fixed.py

echo "✅ DeepResonance Hierarchical Training (1-epoch fix v3) Complete: $(date)"
