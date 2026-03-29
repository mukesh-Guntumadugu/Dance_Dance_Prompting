#!/bin/bash
#SBATCH --job-name=all_models_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/all_models_test_%j.out

echo "=============================================="
echo "  ALL MODELS BENCHMARK — Bad Ketchup"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node:   $SLURM_NODELIST"
echo "  Start:  $(date)"
echo "=============================================="

cd /data/mg546924/llm_beatmap_generator

# Use the deepresonance env as the orchestrator (has librosa, torch, etc.)
/data/mg546924/conda_envs/deepresonance_env/bin/python test_all_models_bad_ketchup.py

echo ""
echo "Finished: $(date)"
