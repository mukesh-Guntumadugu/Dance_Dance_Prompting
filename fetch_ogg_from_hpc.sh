#!/bin/bash

# This script pulls all missing .ogg files from the HPC cluster down to your laptop
# maintaining the same directory structure.
# You will be prompted to enter your HPC password.

echo "Connecting to hpc.ent.ohio.edu to fetch missing .ogg files..."
rsync -avzm -e "ssh -o StrictHostKeyChecking=no" --include="*/" --include="*.ogg" --exclude="*" \
  mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/ \
  /Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/

echo "Done!"
