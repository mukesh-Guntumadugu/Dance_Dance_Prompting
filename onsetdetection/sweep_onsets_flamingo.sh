#!/bin/bash
#SBATCH --job-name=flamingo_sweep
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/flamingo_sweep_%j.out

# Array of segment durations to sweep over
CHUNK_SIZES=(20 15 10 5)

echo "================================================----"
echo "Starting Sequential Onset Extraction Sweep For Flamingo"
echo "================================================----"

cd /data/mg546924/llm_beatmap_generator
export PYTHONUNBUFFERED=1
export BENCHMARK_PROJ=/data/mg546924/llm_beatmap_generator
export HF_HOME=/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints
export LD_LIBRARY_PATH=/data/mg546924/conda_envs/deepresonance_env/lib/python3.10/site-packages/nvidia/cusparse/lib:/data/mg546924/conda_envs/deepresonance_env/lib:$LD_LIBRARY_PATH

for chunk in "${CHUNK_SIZES[@]}"; do
    echo ""
    echo "▶️ Running Flamingo with chunk size: ${chunk}s"
    echo "------------------------------------------------"
    
    /data/mg546924/music_flamingo_env/bin/python -u onsetdetection/extract_onsets_flamingo.py --chunk_sec "$chunk"
    
    echo "✅ Completed Flamingo chunk size: ${chunk}s"
done

echo ""
echo "🎉 All Flamingo sweeps completed successfully!"
