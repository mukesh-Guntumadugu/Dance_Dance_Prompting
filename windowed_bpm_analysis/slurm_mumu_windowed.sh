#!/bin/bash
#SBATCH --job-name=mumu_win_bpm
#SBATCH --output=mumu_win_bpm_%j.log
#SBATCH --error=mumu_win_bpm_%j.err
#SBATCH --partition=defq
#SBATCH --exclusive
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00

cd /data/mg546924/llm_beatmap_generator/windowed_bpm_analysis
source /data/mg546924/miniconda3/bin/activate qwenenv

echo "Running MuMu (C0: Stateless Chunk)"
python -u compute_windowed_rmse.py --model MuMu --mode stateless_chunk

echo "Running MuMu (C1: True History)"
python -u compute_windowed_rmse.py --model MuMu --mode true_history

echo "Running MuMu (C2: Fake History)"
python -u compute_windowed_rmse.py --model MuMu --mode fake_history

echo "Running MuMu (C3: Full Song)"
python -u compute_windowed_rmse.py --model MuMu --mode full_song
