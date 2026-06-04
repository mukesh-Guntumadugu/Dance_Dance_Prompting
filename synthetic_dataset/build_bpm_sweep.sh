#!/bin/bash
mkdir -p sweep_dataset
echo "song_name,duration,category,bpm_sequence" > sweep_dataset/sweep_summary.csv

echo "Generating BPM sweep from 60 to 240..."
for bpm in {60..240}; do
    python3 generate_fixed_bpm.py --bpm $bpm --output_dir sweep_dataset/bpm_$bpm --csv_summary sweep_dataset/sweep_summary.csv
done
echo "Done generating sweep dataset!"
