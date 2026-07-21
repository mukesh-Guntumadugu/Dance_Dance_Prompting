#!/bin/bash
#SBATCH --job-name=swp_Dee_wav
#SBATCH --output=DeepResonance_sweep_wav_%j.log
#SBATCH --error=DeepResonance_sweep_wav_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodelist=node009
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=defq

cd /data/mg546924/llm_beatmap_generator/synthetic_dataset
source /data/mg546924/miniconda3/bin/activate /data/mg546924/conda_envs/deepresonance_env

echo "Resuming DeepResonance on sweep dataset (wav format)..."
python -u evaluate_sweep.py --model DeepResonance --mode full_song --ext wav
