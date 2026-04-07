#!/bin/bash
#SBATCH --job-name=qwen_onset_sweep
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=logs/qwen_sweep_%j.out
#SBATCH --error=logs/qwen_sweep_%j.err

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

mkdir -p logs

# 1. Activate Python environment
CONDA_BIN="/data/mg546924/conda_envs/qwenenv/bin"
if [ -f "$CONDA_BIN/python" ]; then
    echo "Using conda env: $CONDA_BIN"
    export PATH="$CONDA_BIN:$PATH"
    export PYTHONNOUSERSITE=1
else
    echo "❌ Conda env not found at $CONDA_BIN — aborting."
    exit 1
fi

cd /data/mg546924/llm_beatmap_generator
export PYTHONPATH="/data/mg546924/llm_beatmap_generator:$PYTHONPATH"

# 2. Start Qwen Server
SERVER_PORT=8000
MODEL_DIR="/data/mg546924/models/Qwen2-Audio-7B-Instruct"

echo ""
echo "=== Starting Qwen2-Audio Server (port $SERVER_PORT) ==="
python3 src/hpc_qwen_server.py \
    --model-dir "$MODEL_DIR" \
    --lora-dir "/data/mg546924/models/qwen2-audio-lora-onsets" \
    --host 0.0.0.0 \
    --port $SERVER_PORT &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

echo "Waiting for server to become healthy..."
for i in $(seq 1 180); do
    sleep 5
    STATUS=$(curl -s http://localhost:$SERVER_PORT/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('model_loaded','false'))" 2>/dev/null)
    if [ "$STATUS" = "True" ]; then
        echo "✅ Server is ready! (waited ${i}x5s)"
        break
    fi
done

if [ "$STATUS" != "True" ]; then
    echo "❌ Server failed to start within 15 minutes. Check logs."
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# 3. Run the Sequential Onset Sweeps
echo ""
echo "=== Starting Onset Chunk Sweeps ==="
bash sweep_onsets_sequential.sh

# 4. Cleanup
echo ""
echo "=== Sweep complete. Shutting down server. ==="
kill $SERVER_PID 2>/dev/null
echo "Job ended: $(date)"
