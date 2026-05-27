#!/bin/bash
#SBATCH --job-name=qwen_win_bpm
#SBATCH --output=qwen_win_bpm_%j.log
#SBATCH --error=qwen_win_bpm_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00

cd /data/mg546924/llm_beatmap_generator/windowed_bpm_analysis
source /data/mg546924/miniconda3/bin/activate qwenenv

echo "Running Qwen (C0: Stateless Chunk)"
python compute_windowed_rmse.py --model Qwen --mode stateless_chunk

echo "Running Qwen (C1: True History)"
python compute_windowed_rmse.py --model Qwen --mode true_history

echo "Running Qwen (C2: Fake History)"
python compute_windowed_rmse.py --model Qwen --mode fake_history

echo "Running Qwen (C3: Full Song)"
python compute_windowed_rmse.py --model Qwen --mode full_song
