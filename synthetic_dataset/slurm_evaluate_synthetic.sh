#!/bin/bash
#SBATCH --job-name=eval_synthetic
#SBATCH --output=eval_synthetic_%j.log
#SBATCH --error=eval_synthetic_%j.err
#SBATCH --partition=defq
#SBATCH --gres=gpu:A6000:1
#SBATCH --time=24:00:00

# 1. Navigate to the synthetic dataset directory on the HPC
cd /data/mg546924/llm_beatmap_generator/synthetic_dataset

# 2. Activate the primary conda environment to run the evaluation script
source /data/mg546924/miniconda3/bin/activate qwenenv

echo "====================================="
echo "Evaluating on Synthetic Dataset V2"
echo "====================================="

# You can change "--mode" to stateless_chunk, true_history, fake_history, or full_song
# You can change "--ext" to wav, ogg, or mp3

# 1. Run Librosa Baseline
echo "Testing Librosa Baseline..."
python -u evaluate_on_synthetic.py --model Librosa --mode stateless_chunk --ext wav

# 2. Run Qwen
echo "Testing Qwen..."
python -u evaluate_on_synthetic.py --model Qwen --mode stateless_chunk --ext wav

# 3. Run MuMu
echo "Testing MuMu..."
python -u evaluate_on_synthetic.py --model MuMu --mode stateless_chunk --ext wav

# 4. Run Flamingo
echo "Testing Flamingo..."
python -u evaluate_on_synthetic.py --model Flamingo --mode stateless_chunk --ext wav

# 5. Run DeepResonance
echo "Testing DeepResonance..."
python -u evaluate_on_synthetic.py --model DeepResonance --mode stateless_chunk --ext wav

echo "====================================="
echo "All Evaluations Complete!"
