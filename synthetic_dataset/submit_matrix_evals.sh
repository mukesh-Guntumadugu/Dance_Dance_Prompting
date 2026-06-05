#!/bin/bash
#SBATCH --job-name=matrix_eval
#SBATCH --output=logs/matrix_eval_%A_%a.log
#SBATCH --error=logs/matrix_eval_%A_%a.err
#SBATCH --array=60-240
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=defq

# Usage: sbatch --export=MODEL=Qwen submit_matrix_evals.sh
# Expected MODEL values: Qwen, Flamingo, MuMu, Librosa, DeepResonance

if [ -z "$MODEL" ]; then
    echo "Error: MODEL environment variable must be set (e.g., --export=MODEL=Qwen)"
    exit 1
fi

BASE_BPM=$SLURM_ARRAY_TASK_ID

mkdir -p logs

echo "Evaluating Model: $MODEL on Base BPM: $BASE_BPM"

export HF_HOME=/data/mg546924/hf_cache
export TORCH_HOME=/data/mg546924/hf_cache

cd /data/mg546924/llm_beatmap_generator
source /data/mg546924/miniconda3/bin/activate base

# Activate specific environment based on model
case "$MODEL" in
    "Qwen")
        conda activate /data/mg546924/conda_envs/qwenenv
        ;;
    "Flamingo")
        conda activate /data/mg546924/music_flamingo_env
        ;;
    "MuMu")
        conda activate /data/mg546924/miniconda3/envs/mumullama
        ;;
    "DeepResonance")
        conda activate /data/mg546924/conda_envs/deepresonance_env
        ;;
    "Librosa")
        # Librosa works in almost any env, use base or qwenenv
        conda activate /data/mg546924/conda_envs/qwenenv
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

# Execute without unbuffered stdout to let Python output normally
python -u synthetic_dataset/evaluate_matrix.py --model $MODEL --base_bpm $BASE_BPM --matrix_dir matrix_dataset

echo "Evaluation finished for $MODEL on Base BPM $BASE_BPM"
