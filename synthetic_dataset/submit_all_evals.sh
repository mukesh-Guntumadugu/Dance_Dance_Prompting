#!/bin/bash
echo "Canceling old monolithic job..."
scancel 32141
scancel 32142

echo "Submitting 5 individual parallel jobs..."
sbatch --nodelist=node001 slurm_eval_librosa.sh
sbatch --nodelist=node004 slurm_eval_qwen.sh
sbatch --nodelist=node007 slurm_eval_mumu.sh
sbatch --nodelist=node008 slurm_eval_flamingo.sh
sbatch --nodelist=node009 slurm_eval_deepresonance.sh

echo "All jobs submitted in parallel!"
