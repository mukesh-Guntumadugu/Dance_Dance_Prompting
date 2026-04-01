#!/bin/bash
#SBATCH --job-name=dr_measure
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/dr_measure_%j.out

echo "=== DEEPRESONANCE MEASURE-BY-MEASURE: Bad Ketchup === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1
export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator
export BENCHMARK_AUDIO="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg"
export BENCHMARK_OUT="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup"

# Required for DeepResonance bitsandbytes CUDA library
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

echo "Running target script..."
/data/mg546924/conda_envs/deepresonance_env/bin/python -u scripts/DeepResonance/deepresonance_measure_generator.py

echo "=== DONE $(date) ==="
