#!/bin/bash
#SBATCH --job-name=eval_flamingo
#SBATCH --output=eval_flamingo_%j.log
#SBATCH --error=eval_flamingo_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00

cd /data/mg546924/llm_beatmap_generator/synthetic_dataset
source /data/mg546924/miniconda3/bin/activate qwenenv

echo "Testing Flamingo..."
python -u evaluate_on_synthetic.py --model Flamingo --mode stateless_chunk --ext wav
