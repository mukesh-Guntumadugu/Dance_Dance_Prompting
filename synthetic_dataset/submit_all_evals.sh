#!/bin/bash
echo "Canceling old monolithic job..."
scancel 32141
scancel 32142

echo "Submitting 5 individual parallel jobs..."
sbatch --nodelist=node001 slurm_eval_librosa.sh
sbatch --nodelist=node002 slurm_eval_qwen.sh
sbatch --nodelist=node003 slurm_eval_mumu.sh
sbatch --nodelist=node005 slurm_eval_flamingo.sh
sbatch --nodelist=node006 slurm_eval_deepresonance.sh

echo "All jobs submitted in parallel!"
