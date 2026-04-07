#!/bin/bash
#SBATCH --job-name=flam_onset_sweep
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/flamingo_sweep_%j.out
#SBATCH --error=logs/flamingo_sweep_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p logs

# 1. Activate Python environment
CONDA_BIN="/data/mg546924/conda_envs/flamingo_env/bin"
if [ -f "$CONDA_BIN/python" ]; then
    echo "Using conda env: $CONDA_BIN"
    export PATH="$CONDA_BIN:$PATH"
    export PYTHONUNBUFFERED=1
else
    echo "❌ Conda env not found at $CONDA_BIN — aborting."
    exit 1
fi

cd /data/mg546924/llm_beatmap_generator
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

# 2. Run the Sequential Onset Sweeps
echo ""
echo "=== Starting Flamingo Onset Chunk Sweeps ==="
bash sweep_onsets_flamingo.sh

# 3. Cleanup
echo ""
echo "=== Sweep complete. ==="
echo "Job ended: $(date)"
