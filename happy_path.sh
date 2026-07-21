#!/bin/bash
# ==============================================================================
# happy_path.sh
# 
# End-to-End Orchestrator for Rhythmic Beatmap Generation across all 4 models.
# Usage:
#   bash happy_path.sh /path/to/my_new_song.wav
# ==============================================================================

set -e

if [ -z "$1" ]; then
    echo "❌ Error: Please provide an audio file."
    echo "Usage: bash happy_path.sh <path_to_audio_file>"
    exit 1
fi

AUDIO_FILE="$1"

if [ ! -f "$AUDIO_FILE" ]; then
    echo "❌ Error: File not found -> $AUDIO_FILE"
    exit 1
fi

SONG_NAME=$(basename "$AUDIO_FILE" | cut -f 1 -d '.')
OUT_DIR="outputs/happy_path/${SONG_NAME}"
mkdir -p "$OUT_DIR"

echo "========================================================="
echo "🎵 HAPP PATH: Full Back-to-Back Pipeline Initializing 🎵"
echo "========================================================="
echo "🎤 Input Song : $SONG_NAME"
echo "📂 Output Dir : $OUT_DIR"
echo "========================================================="

# 1. Extract BPM using a fast inline Python script
echo -n "⏳ [1/2] Analyzing Audio & Extracting BPM... "

BPM=$(python3 -c "
import librosa
import sys
import warnings
warnings.filterwarnings('ignore')
try:
    y, sr = librosa.load('$AUDIO_FILE')
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa beat_track can return an array or float depending on version
    bpm = tempo[0] if hasattr(tempo, '__len__') else tempo
    print(round(bpm, 2))
except Exception as e:
    print(130.0) # Fallback
")

echo "Done! (Detected BPM: $BPM)"
echo "---------------------------------------------------------"

# 2. Run Generators
echo "🚀 [2/2] Running Hierarchical Models..."
echo ""

PYTHON_ENV="/data/mg546924/conda_envs/qwenenv/bin/python"
if [ ! -f "$PYTHON_ENV" ]; then
    PYTHON_ENV="python3" # Fallback to local python if not on HPC
fi

echo "🤖 1. MuMu-LLaMA..."
$PYTHON_ENV scripts/generate_hierarchical_beatmap.py \
    --audio "$AUDIO_FILE" \
    --bpm "$BPM" \
    --difficulty "Challenge" \
    --out "$OUT_DIR/${SONG_NAME}_mumu.ssc" || echo "⚠️ MuMu-LLaMA generation failed."

echo ""
echo "🤖 2. Qwen2-Audio..."
$PYTHON_ENV scripts/generate_hierarchical_beatmap_qwen.py \
    --audio "$AUDIO_FILE" \
    --bpm "$BPM" \
    --difficulty "Challenge" \
    --out "$OUT_DIR/${SONG_NAME}_qwen.ssc" || echo "⚠️ Qwen2-Audio generation failed."

echo ""
echo "🤖 3. Music-Flamingo..."
$PYTHON_ENV scripts/generate_hierarchical_beatmap_flamingo.py \
    --audio "$AUDIO_FILE" \
    --bpm "$BPM" \
    --difficulty "Challenge" \
    --out "$OUT_DIR/${SONG_NAME}_flamingo.ssc" || echo "⚠️ Music-Flamingo generation failed."

echo ""
echo "🤖 4. DeepResonance..."
$PYTHON_ENV scripts/generate_hierarchical_beatmap_deepresonance.py \
    --audio "$AUDIO_FILE" \
    --bpm "$BPM" \
    --difficulty "Challenge" \
    --out "$OUT_DIR/${SONG_NAME}_deepresonance.ssc" || echo "⚠️ DeepResonance generation failed."

echo "========================================================="
echo "✅ HAPPY PATH COMPLETE!"
echo "Generated Beatmaps are available in: $OUT_DIR/"
echo "========================================================="
