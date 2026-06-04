#!/bin/bash
# Submit SLURM jobs for evaluating models on the sweep dataset

MODELS=("Qwen" "Flamingo" "MuMu")
FORMATS=("wav" "mp3" "ogg")
MODE="full_song"

for MODEL in "${MODELS[@]}"; do
    for EXT in "${FORMATS[@]}"; do
        
        JOB_NAME="swp_${MODEL:0:3}_${EXT}"
        
        if [ "$MODEL" == "Flamingo" ]; then
            ENV_NAME="flamingo_env"
        elif [ "$MODEL" == "DeepResonance" ]; then
            ENV_NAME="deepresonance_env"
        else
            ENV_NAME="qwenenv"
        fi
        
        # Create a temporary slurm script
        cat <<EOF > tmp_submit.slurm
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=${MODEL}_sweep_${EXT}_%j.log
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=defq

cd /data/mg546924/llm_beatmap_generator
source /data/mg546924/miniconda3/bin/activate $ENV_NAME

echo "Running $MODEL on sweep dataset ($EXT format)..."
python evaluate_sweep.py --model $MODEL --mode $MODE --ext $EXT
EOF

        sbatch tmp_submit.slurm
        rm tmp_submit.slurm
        
    done
done

echo "All sweep evaluation jobs submitted!"
