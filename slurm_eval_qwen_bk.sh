#!/bin/bash
#SBATCH --job-name=eval_qwen
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-01:00:00
#SBATCH --output=/data/mg546924/llm_beatmap_generator/logs/eval_qwen_%j.log

cd /data/mg546924/llm_beatmap_generator
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"
/data/mg546924/conda_envs/qwenenv/bin/python scripts/evaluate_qwen_bad_ketchup.py
