#!/bin/bash
#SBATCH --job-name=extract_latents
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/extract_latents_%j.out

PROJ_DIR="/data/mg546924/llm_beatmap_generator"
AUDIO_FILE="$PROJ_DIR/test_120bpm.wav"
OUT_DIR="$PROJ_DIR/latent_outputs"

mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/chunks"

# 1. Chop the audio into 1-second chunks (assuming 10s file, we get 10 chunks)
# (In bash, we can use ffmpeg, or we can assume a python script does this. Let's do it in python briefly inside the loop or beforehand)

cat << 'EOF' > "$OUT_DIR/chop.py"
import soundfile as sf
import sys
import numpy as np

y, sr = sf.read(sys.argv[1])
chunk_len = sr * 1
for i in range(10):
    chunk = y[i*chunk_len:(i+1)*chunk_len]
    sf.write(f"{sys.argv[2]}/chunk_{i:02d}.wav", chunk, sr)
EOF

python "$OUT_DIR/chop.py" "$AUDIO_FILE" "$OUT_DIR/chunks"

# 2. Extract for each chunk and each model
for i in $(seq -w 0 9); do
    CHUNK_FILE="$OUT_DIR/chunks/chunk_${i}.wav"
    
    # Qwen
    /data/mg546924/conda_envs/qwenenv/bin/python "$PROJ_DIR/hpc_extract_windowed_latents.py" \
        --model qwen \
        --audio "$CHUNK_FILE" \
        --out "$OUT_DIR/qwen_chunk_${i}.pkl"
        
    # MuMu
    /data/mg546924/conda_envs/mumumv/bin/python "$PROJ_DIR/hpc_extract_windowed_latents.py" \
        --model mumu \
        --audio "$CHUNK_FILE" \
        --out "$OUT_DIR/mumu_chunk_${i}.pkl"
        
    # EnCodec (can run in base env)
    python "$PROJ_DIR/hpc_extract_windowed_latents.py" \
        --model encodec \
        --audio "$CHUNK_FILE" \
        --out "$OUT_DIR/encodec_chunk_${i}.pkl"
done

echo "Done! Data saved to $OUT_DIR"
