#!/bin/bash

# This script pulls all the Qwen Sweep CSV output files from the HPC cluster 
# down to your laptop maintaining the same directory structure.
# You will be prompted to enter your HPC password.

echo "Connecting to hpc.ent.ohio.edu to fetch the newest CSV sweep data..."
rsync -avzm -e "ssh -o StrictHostKeyChecking=no" --include="*/" --include="Bad_Ketchup_Qwen_*.csv" --exclude="*" \
  mg546924@hpc.ent.ohio.edu:"/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's\ Arrow\ Arrangements/Bad\ Ketchup/qwen_onsets/" \
  "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/qwen_onsets/"

echo "Done! The files are now synced locally and ready for evaluation."
