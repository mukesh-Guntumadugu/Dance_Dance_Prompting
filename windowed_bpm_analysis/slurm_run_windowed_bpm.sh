#!/bin/bash
#SBATCH --job-name=windowed_rmse
#SBATCH --output=windowed_rmse_%A_%a.log
#SBATCH --error=windowed_rmse_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --array=0-3

echo "Starting Windowed BPM RMSE Evaluation..."
date

cd /data/mg546924/llm_beatmap_generator/windowed_bpm_analysis

# Use base environment or an environment that has librosa, soundfile, pandas, numpy
source /data/mg546924/miniconda3/bin/activate qwenenv

MODEL="Qwen" # Change this to test other models (MuMu, Flamingo, DeepResonance)
MODES=("stateless_chunk" "true_history" "fake_history" "full_song")
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

echo "Running $MODEL in mode: $MODE"
python compute_windowed_rmse.py --model $MODEL --mode $MODE

echo "Finished Evaluation!"
date
