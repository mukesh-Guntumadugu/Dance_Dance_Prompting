#!/bin/bash
#SBATCH --job-name=test_qwen
#SBATCH --output=test_qwen.log
#SBATCH --error=test_qwen.err
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=defq

cd /data/mg546924/llm_beatmap_generator
source /data/mg546924/miniconda3/bin/activate qwenenv

echo "Running Qwen Test..."
cat <<EOF > test_qwen_script.py
import os, sys
print("Starting Qwen import...")
sys.path.insert(0, "/data/mg546924/llm_beatmap_generator")
from src.qwen_interface import setup_qwen
print("Imported successfully! Calling setup_qwen...")
setup_qwen()
print("Setup complete!")
EOF

python -u test_qwen_script.py
echo "Exit code: $?"
