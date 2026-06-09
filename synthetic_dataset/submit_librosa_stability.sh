#!/bin/bash
# Submit SLURM jobs to test Librosa stability across 10 iterations for each format

FORMATS=("wav" "mp3" "ogg")
ITERATIONS=10

for FORMAT in "${FORMATS[@]}"; do
    
    # Create a temporary slurm script for this format
    cat <<EOF > tmp_librosa_stability.slurm
#!/bin/bash
#SBATCH --job-name=Lib_stab_${FORMAT}
#SBATCH --output=Librosa_stability_${FORMAT}_%j.log
#SBATCH --error=Librosa_stability_${FORMAT}_%j.err
#SBATCH --time=12:00:00
#SBATCH --nodelist=node001
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=defq

# Navigate to the dataset directory
cd /data/mg546924/llm_beatmap_generator/synthetic_dataset

echo "Running Librosa stability test for format: ${FORMAT}..."

for RUN_IDX in {1..${ITERATIONS}}; do
    echo "Starting Iteration \$RUN_IDX..."
    /data/mg546924/conda_envs/qwenenv/bin/python evaluate_sweep.py --model Librosa --mode full_song --ext ${FORMAT} --run_idx \$RUN_IDX
    echo "Finished Iteration \$RUN_IDX."
done

echo "All ${ITERATIONS} iterations for ${FORMAT} complete!"
EOF

    # Submit the job
    sbatch tmp_librosa_stability.slurm
done

# Clean up
rm tmp_librosa_stability.slurm

echo "All 3 Librosa stability jobs submitted (each running ${ITERATIONS} iterations)!"
