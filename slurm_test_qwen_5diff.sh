#!/bin/bash
#SBATCH --job-name=qwen_5diff
#SBATCH --output=logs/qwen_5diff_%j.log
#SBATCH --error=logs/qwen_5diff_%j.log
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=defq
#SBATCH --gres=gpu:1

set -e
cd /data/mg546924/llm_beatmap_generator
mkdir -p outputs logs

export CUDA_VISIBLE_DEVICES=0
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/data/mg546924/conda_envs/qwenenv/bin/python
SCRIPT=scripts/test_qwen_all_difficulties.py

echo "=============================================="
echo "  Qwen2-Audio Director — All 5 Difficulties"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  Start  : $(date)"
echo "=============================================="

echo ""
echo "Song 1: Bad Ketchup"
$PYTHON -u $SCRIPT \
    --audio "src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg" \
    --bpm 180.0 \
    --out "outputs/qwen_5diff_Bad_Ketchup.ssc"

echo ""
echo "Song 2: Springtime"
$PYTHON -u $SCRIPT \
    --audio "src/musicForBeatmap/Springtime/Kommisar - Springtime.mp3" \
    --bpm 145.0 \
    --out "outputs/qwen_5diff_Springtime.ssc"

echo ""
echo "Song 3: Mecha-Tribe Assault"
$PYTHON -u $SCRIPT \
    --audio "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ogg" \
    --bpm 200.0 \
    --out "outputs/qwen_5diff_MechaTribe.ssc"

echo ""
echo "=============================================="
echo "[OK] Qwen 5-Difficulty Generation Complete!"
echo "   End : $(date)"
echo "=============================================="
