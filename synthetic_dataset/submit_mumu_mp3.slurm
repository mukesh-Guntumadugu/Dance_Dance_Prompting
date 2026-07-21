#!/bin/bash
#SBATCH --job-name=swp_MuM_mp3
#SBATCH --output=MuMu_sweep_mp3_%j.log
#SBATCH --error=MuMu_sweep_mp3_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:A6000:1
#SBATCH --nodelist=node001
#SBATCH --partition=defq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

cd /data/mg546924/llm_beatmap_generator/synthetic_dataset
source /data/mg546924/miniconda3/bin/activate /home/mg546924/.conda/envs/mumullama

echo "Running MuMu on sweep dataset (mp3 format)..."
python -u evaluate_sweep.py --model MuMu --mode full_song --ext mp3
