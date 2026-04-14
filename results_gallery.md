# Pattern Finding & Beatmap Results

Here are the visual charts generated from our recent `slurm_run_pattern_finding.sh` and related scripts.
*(If you are in VS Code, right-click this file and select **Open Preview**, or click the preview icon in the top right).*

## 1. Macro Cluster Transitions (Markov Chain)
This heatmap shows the transition probabilities between different structural "moods" (clusters) in the beatmaps.
![Macro Transitions](/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/pattern_finding_results/macro_cluster_transitions_heatmap.png)

## 2. Micro Markov Transitions (Note-to-Note)
Detailed note sequence transitions showing the localized rhythmic style of the dataset.
![Micro Transitions](/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/pattern_finding_results/micro_markov_transitions_heatmap.png)

## 3. Audio & Stepmania Correlation
This demonstrates the mapping between extracted audio features (via Librosa) and the high-dimensional structural representations from the Stepmania files.
![Audio vs Stepmania](/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/pattern_finding_results/audio_stepmania_correlation.png)
