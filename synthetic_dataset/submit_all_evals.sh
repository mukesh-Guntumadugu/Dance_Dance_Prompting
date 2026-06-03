#!/bin/bash
echo "Canceling old monolithic job..."
scancel 32141
scancel 32142

echo "Submitting 5 individual parallel jobs..."
sbatch slurm_eval_librosa.sh
sbatch slurm_eval_qwen.sh
sbatch slurm_eval_mumu.sh
sbatch slurm_eval_flamingo.sh
sbatch slurm_eval_deepresonance.sh

echo "All jobs submitted in parallel!"
