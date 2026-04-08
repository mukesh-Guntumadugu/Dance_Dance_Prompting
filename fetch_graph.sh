#!/bin/bash

echo "Connecting to hpc.ent.ohio.edu to pull the Graphical Map!..."
rsync -avzm -e "ssh -o StrictHostKeyChecking=no" \
  mg546924@hpc.ent.ohio.edu:/data/mg546924/llm_beatmap_generator/analysis_reports/Qwen_Onset_Sweep_Comparison_Bad_Ketchup.png \
  /Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/

echo "Done! Check your root folder for the PNG image."
