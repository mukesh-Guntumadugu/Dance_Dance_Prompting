#!/bin/bash
#SBATCH --job-name=qwen_sft_goauto
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00           
#SBATCH --output=logs/qwen_sft_goauto_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

echo "=== QWEN SFT HIERARCHICAL: Evaluating goAutoStepper Dataset === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

PYTHON_ENV="/data/mg546924/conda_envs/qwenenv/bin/python"
BASE_DIR="/data/mg546924/llm_beatmap_generator/synthetic_dataset/out_GoautoStepper"

find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r SONG_DIR; do
    SONG_NAME=$(basename "$SONG_DIR")
    AUDIO_FILE=$(find "$SONG_DIR" -maxdepth 1 -type f -name "*.ogg" -o -name "*.mp3" -o -name "*.wav" | head -n 1)

    if [ -z "$AUDIO_FILE" ]; then
        echo "No audio file found for $SONG_NAME, skipping."
        continue
    fi

    # Extract BPM fallback
    BPM=$($PYTHON_ENV -c "
import librosa, warnings
warnings.filterwarnings('ignore')
try:
    y, sr = librosa.load('$AUDIO_FILE')
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    bpm = tempo[0] if hasattr(tempo, '__len__') else tempo
    print(round(bpm, 2))
except:
    print(130.0)
")

    echo "--------------------------------------------------------"
    echo "Processing Song: $SONG_NAME | BPM: $BPM"
    echo "Audio: $AUDIO_FILE"
    echo "--------------------------------------------------------"

    OUT_DIR="$SONG_DIR/Qwen_SFT"
    mkdir -p "$OUT_DIR"

    # Run for Challenge difficulty (1 iteration)
    $PYTHON_ENV scripts/generate_hierarchical_beatmap_qwen.py \
        --audio "$AUDIO_FILE" \
        --bpm "$BPM" \
        --difficulty "Challenge" \
        --out "$OUT_DIR/${SONG_NAME}_qwen_sft.ssc"

done

echo "=== EVALUATION COMPLETE $(date) ==="
