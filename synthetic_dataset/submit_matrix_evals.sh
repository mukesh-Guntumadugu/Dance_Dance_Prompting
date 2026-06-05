#!/bin/bash
#SBATCH --job-name=matrix_eval
#SBATCH --output=logs/matrix_eval_%j.log
#SBATCH --error=logs/matrix_eval_%j.err
#SBATCH --time=72:00:00
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

mkdir -p logs

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
        conda activate /home/mg546924/.conda/envs/mumullama
        ;;
    "DeepResonance")
        conda activate /data/mg546924/conda_envs/deepresonance_env
        ;;
    "Librosa")
        conda activate /data/mg546924/conda_envs/qwenenv
        ;;
    *)
        echo "Unknown model: $MODEL"
        exit 1
        ;;
esac

echo "Starting sequential evaluation loop for Model: $MODEL"

for BASE_BPM in {60..240}; do
    OUT_CSV="matrix_dataset/${MODEL}_base_bpm_${BASE_BPM}_rmse.csv"
    
    # Check if this CSV already exists and has size > 0 to avoid re-running
    if [ -s "$OUT_CSV" ]; then
        echo "Skipping Base BPM $BASE_BPM for $MODEL (already generated)"
        continue
    fi
    
    echo "Evaluating Model: $MODEL on Base BPM: $BASE_BPM"
    python -u synthetic_dataset/evaluate_matrix.py --model $MODEL --base_bpm $BASE_BPM --matrix_dir matrix_dataset
    echo "Completed Base BPM $BASE_BPM for $MODEL"
    echo "---------------------------------------------------"
done

echo "Evaluation fully finished for $MODEL"
