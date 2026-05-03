#!/bin/bash
#SBATCH --job-name=rhythm_analysis
#SBATCH --output=logs/rhythm_analysis_%j.log
#SBATCH --error=logs/rhythm_analysis_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --partition=short

# Load your environment if needed
# source /path/to/your/conda/etc/profile.d/conda.sh
# conda activate your_env

echo "Starting Rhythm and Tempo Analysis..."

python3 scripts/extract_rhythm_density.py
python3 scripts/categorize_tempos.py
python3 scripts/analyze_local_bpm.py

echo "Analysis Complete!"
