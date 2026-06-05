#!/bin/bash
#SBATCH --job-name=matrix_gen
#SBATCH --output=logs/matrix_gen_%A_%a.log
#SBATCH --error=logs/matrix_gen_%A_%a.err
#SBATCH --array=60-240
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=defq

# %A is the array job ID, %a is the array task ID (the base BPM).
BASE_BPM=$SLURM_ARRAY_TASK_ID

mkdir -p matrix_dataset
mkdir -p logs

echo "Generating matrix dataset for Base BPM: $BASE_BPM"

cd /data/mg546924/llm_beatmap_generator
source /data/mg546924/miniconda3/bin/activate qwenenv

python synthetic_dataset/generate_matrix_dataset.py --base_bpm $BASE_BPM --output_dir matrix_dataset

echo "Finished generating Base BPM: $BASE_BPM"
