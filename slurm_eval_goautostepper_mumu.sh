#!/bin/bash
#SBATCH --job-name=mumu_goauto
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00           
#SBATCH --output=logs/mumu_goauto_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

echo "=== MUMU MEASURE-BY-MEASURE: Evaluating goAutoStepper Dataset === $(date)"
cd /data/mg546924/llm_beatmap_generator

export PYTHONUNBUFFERED=1

PYTHON_DRIVER="/data/mg546924/conda_envs/qwenenv/bin/python"
BASE_DIR="/data/mg546924/llm_beatmap_generator/synthetic_dataset/out_GoautoStepper"

for ITERATION in {1..1}; do
    echo "========================================================"
    echo "=== STARTING ITERATION $ITERATION / 1 === $(date)"
    echo "========================================================"

    find "$BASE_DIR" -mindepth 1 -maxdepth 1 -type d | sort | while read -r SONG_DIR; do
        SONG_NAME=$(basename "$SONG_DIR")
        
        # Locate the audio file (.ogg, .mp3, .wav)
        AUDIO_FILE=$(find "$SONG_DIR" -maxdepth 1 -type f -name "*.ogg" -o -name "*.mp3" -o -name "*.wav" | head -n 1)

        if [ -z "$AUDIO_FILE" ]; then
            echo "No audio file found for $SONG_NAME, skipping."
            continue
        fi

        echo "--------------------------------------------------------"
        echo "Iteration: $ITERATION | Processing Song: $SONG_NAME"
        echo "Audio: $AUDIO_FILE"
        echo "--------------------------------------------------------"

        export BENCHMARK_AUDIO="$AUDIO_FILE"
        export BENCHMARK_OUT="$SONG_DIR"
        export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator

        # Run the generator
        $PYTHON_DRIVER -u scripts/Mumu/mumu_measure_generator.py

    done

    echo "=== ITERATION $ITERATION DONE $(date) ==="
done

echo "=== EVALUATION COMPLETE $(date) ==="
