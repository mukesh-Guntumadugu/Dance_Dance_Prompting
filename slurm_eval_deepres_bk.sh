#!/bin/bash
#SBATCH --job-name=eval_bk_dr
#SBATCH --output=logs/eval_bk_dr_%j.out
#SBATCH --error=logs/eval_bk_dr_%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL

set -e
cd /data/mg546924/llm_beatmap_generator

export PYTHONNOUSERSITE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS=ignore

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

/data/mg546924/conda_envs/qwenenv/bin/python scripts/evaluate_deepres_bad_ketchup.py

echo "Job $SLURM_JOB_ID done at $(date)"
