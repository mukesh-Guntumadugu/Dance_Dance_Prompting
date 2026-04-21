#!/bin/bash
#SBATCH --job-name=tempo_extraction
#SBATCH --partition=defq
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/tempo_extraction_%j.out

cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1
PYTHON=/data/mg546924/conda_envs/deepresonance_env/bin/python

echo "Started: $(date)"
$PYTHON onsetdetection/extract_tempo_changes.py \
    --batch_dir src/musicForBeatmap \
    --window_size 20 \
    --out_csv onsetdetection/Tempo_Change_Analysis.csv

echo "Done: $(date)"
