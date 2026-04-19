#!/bin/bash
#SBATCH --job-name=qwen_grpo_rl
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:30:00
#SBATCH --nodelist=node002
#SBATCH --output=logs/grpo_rl_%j.out
#SBATCH --error=logs/grpo_rl_%j.err

echo "================================================================"
echo "  Qwen2-Audio GRPO RL — Onset Detection"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURMD_NODENAME"
echo "  GPU    : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "  Start  : $(date)"
echo "================================================================"

mkdir -p logs

QWEN_PY=/data/mg546924/conda_envs/qwenenv/bin/python
cd /data/mg546924/llm_beatmap_generator

export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"
export HF_HOME="/data/mg546924/.cache/huggingface"

# Use the large Pixabay 1000-song dataset
export DATASET_OVERRIDE="/data/mg546924/llm_beatmap_generator/sft_dataset_pixabay/dataset.jsonl"

# Optional: warm-start from SFT checkpoint if it exists
# export SFT_CKPT="/data/mg546924/models/qwen2-audio-lora-onsets"

echo ""
echo "Starting GRPO training..."
$QWEN_PY -u scripts/train_qwen2_audio_grpo.py

echo ""
echo "================================================================"
echo "  GRPO Training Finished: $(date)"
echo "================================================================"
