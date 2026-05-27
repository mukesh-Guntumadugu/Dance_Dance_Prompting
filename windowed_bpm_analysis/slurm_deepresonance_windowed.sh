#!/bin/bash
#SBATCH --job-name=deepres_win_bpm
#SBATCH --output=deepres_win_bpm_%j.log
#SBATCH --error=deepres_win_bpm_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:a100:1
#SBATCH --time=24:00:00
#SBATCH --mem=32G

cd /data/mg546924/llm_beatmap_generator/windowed_bpm_analysis
source /data/mg546924/miniconda3/bin/activate deepresonance_env

echo "Running DeepResonance (C0: Stateless Chunk)"
python compute_windowed_rmse.py --model DeepResonance --mode stateless_chunk

echo "Running DeepResonance (C1: True History)"
python compute_windowed_rmse.py --model DeepResonance --mode true_history

echo "Running DeepResonance (C2: Fake History)"
python compute_windowed_rmse.py --model DeepResonance --mode fake_history

echo "Running DeepResonance (C3: Full Song)"
python compute_windowed_rmse.py --model DeepResonance --mode full_song
