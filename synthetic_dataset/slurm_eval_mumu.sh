#!/bin/bash
#SBATCH --job-name=eval_mumu
#SBATCH --output=eval_mumu_%j.log
#SBATCH --error=eval_mumu_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00

cd /data/mg546924/llm_beatmap_generator/synthetic_dataset
source /data/mg546924/miniconda3/bin/activate qwenenv

echo "Testing MuMu (WAV)..."
python -u evaluate_on_synthetic.py --model MuMu --mode stateless_chunk --ext wav

echo "Testing MuMu (MP3)..."
python -u evaluate_on_synthetic.py --model MuMu --mode stateless_chunk --ext mp3

echo "Testing MuMu (OGG)..."
python -u evaluate_on_synthetic.py --model MuMu --mode stateless_chunk --ext ogg
