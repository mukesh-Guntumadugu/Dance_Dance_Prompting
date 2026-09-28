#!/bin/bash
#SBATCH --job-name=mumu_sft_goauto
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00           
#SBATCH --output=logs/mumu_sft_goauto_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

echo "=== MUMU SFT HIERARCHICAL: Evaluating goAutoStepper Dataset === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1

# MuMu env from happy_path.sh
PYTHON_ENV="/data/mg546924/conda_envs/qwenenv/bin/python"
BASE_DIR="/data/mg546924/llm_beatmap_generator/synthetic_dataset/out_GoautoStepper"

find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r SONG_DIR; do
    SONG_NAME=$(basename "$SONG_DIR")
    AUDIO_FILE=$(find "$SONG_DIR" -maxdepth 1 -type f -name "*.ogg" -o -name "*.mp3" -o -name "*.wav" | head -n 1)

    if [ -z "$AUDIO_FILE" ]; then
        continue
    fi

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

    echo "Processing Song: $SONG_NAME | BPM: $BPM"
    OUT_DIR="$SONG_DIR/MuMu_SFT"
    mkdir -p "$OUT_DIR"

    # Run for Challenge difficulty
    $PYTHON_ENV scripts/generate_hierarchical_beatmap.py \
        --audio "$AUDIO_FILE" \
        --bpm "$BPM" \
        --difficulty "Challenge" \
        --out "$OUT_DIR/${SONG_NAME}_mumu_sft.ssc"

done

echo "=== EVALUATION COMPLETE $(date) ==="
