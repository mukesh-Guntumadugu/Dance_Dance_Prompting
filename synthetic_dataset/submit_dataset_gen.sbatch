#!/bin/bash
#SBATCH --job-name=generate_perfect_bpm
#SBATCH --output=generate_perfect_bpm_%j.out
#SBATCH --error=generate_perfect_bpm_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=defq
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

cd $SLURM_SUBMIT_DIR

DATASET_DIR="linear_probing_dataset"
README_FILE="$DATASET_DIR/README.md"

mkdir -p $DATASET_DIR
echo "# Linear Probing Dataset (60-240 BPM)" > $README_FILE
echo "This dataset contains 1-minute stable audio files with identical instrumentation for linear probing." >> $README_FILE
echo "" >> $README_FILE
echo "## Generated Files:" >> $README_FILE

echo "Generating perfect BPM sweep from 60 to 240..."

for bpm in {60..240}; do
    echo "Processing BPM: $bpm"
    python3 generate_perfect_bpm.py --bpm $bpm --duration 60.0 --output_dir $DATASET_DIR
done

echo "Done generating linear probing dataset!"
