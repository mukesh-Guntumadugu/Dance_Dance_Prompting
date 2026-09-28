#!/bin/bash
#SBATCH --job-name=generate_bk
#SBATCH --output=logs/generate_bk_%j.log
#SBATCH --error=logs/generate_bk_%j.log
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --mail-user=mg546924@ohio.edu
#SBATCH --mail-type=END,FAIL

set -e
cd /data/mg546924/llm_beatmap_generator

# Load environment
source /data/mg546924/conda_envs/qwenenv/bin/activate || true

echo "Generating Beatmap for Bad Ketchup using latest DeepResonance weights..."
/data/mg546924/conda_envs/qwenenv/bin/python scripts/generate_hierarchical_beatmap_deepresonance.py \
    --audio "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg" \
    --out "/data/mg546924/llm_beatmap_generator/Bad_Ketchup_DeepResonance.ssc"

echo "Done! Check Bad_Ketchup_DeepResonance.ssc"
