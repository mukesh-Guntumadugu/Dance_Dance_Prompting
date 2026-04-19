#!/bin/bash
#SBATCH --job-name=music_probe
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --nodelist=node002
#SBATCH --output=music_probe_%j.txt

echo "=== Music Knowledge Probe ==="
echo "Job ID: $SLURM_JOB_ID | Node: $SLURM_NODELIST"
echo "============================="

cd /data/mg546924/llm_beatmap_generator
source /home/mg546924/.conda/etc/profile.d/conda.sh

echo ""
echo "--- Probing MuMu-LLaMA ---"
conda run -n mumullama python3 onsetdetection/probe_model_music_knowledge.py --model mumu

echo ""
echo "--- Probing Gemini (text only, no GPU) ---"
conda run -n mumullama python3 onsetdetection/probe_model_music_knowledge.py --model gemini

echo ""
echo "Finished: $(date)"
