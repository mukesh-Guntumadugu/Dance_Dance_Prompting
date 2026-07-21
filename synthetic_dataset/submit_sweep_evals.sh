#!/bin/bash
# Submit SLURM jobs for evaluating models on the sweep dataset

MODELS=("Qwen" "Flamingo" "MuMu" "DeepResonance" "Librosa")
FORMATS=("mp3" "ogg")
MODE="full_song"

for MODEL in "${MODELS[@]}"; do
    for EXT in "${FORMATS[@]}"; do
        
        JOB_NAME="swp_${MODEL:0:3}_${EXT}"
        
        if [ "$MODEL" == "Flamingo" ]; then
            ENV_NAME="/data/mg546924/music_flamingo_env"
            TARGET_NODE="node004"
        elif [ "$MODEL" == "DeepResonance" ]; then
            ENV_NAME="/data/mg546924/conda_envs/deepresonance_env"
            TARGET_NODE="node002"
        elif [ "$MODEL" == "MuMu" ]; then
            ENV_NAME="/home/mg546924/.conda/envs/mumullama"
            TARGET_NODE="node005"
        elif [ "$MODEL" == "Qwen" ]; then
            ENV_NAME="/data/mg546924/conda_envs/qwenenv"
            TARGET_NODE="node006"
        else
            ENV_NAME="/data/mg546924/conda_envs/qwenenv"
            TARGET_NODE="node001"
        fi
        
        # Create a temporary slurm script
        cat <<EOF > tmp_submit.slurm
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=${MODEL}_sweep_${EXT}_%j.log
#SBATCH --error=${MODEL}_sweep_${EXT}_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodelist=$TARGET_NODE
#SBATCH --gres=gpu:A6000:1
#SBATCH --partition=defq
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mg546924@ohio.edu

cd /data/mg546924/llm_beatmap_generator/synthetic_dataset
source /data/mg546924/miniconda3/bin/activate $ENV_NAME

echo "Running $MODEL on sweep dataset ($EXT format)..."
python -u evaluate_sweep.py --model $MODEL --mode $MODE --ext $EXT
EOF

        sbatch tmp_submit.slurm
        rm tmp_submit.slurm
        
    done
done

echo "All sweep evaluation jobs submitted!"
