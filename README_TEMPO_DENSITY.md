# Rhythm and Tempo Analysis Tools

This suite of scripts provides tools for analyzing StepMania beatmap files (.sm and .ssc) to extract rhythmic density and tempo dynamics. These features are critical for training and validating LLM-based beatmap generators (MuMu, Flamingo, Qwen).

## Scripts Overview

### 1. `extract_rhythm_density.py`
- **What it does**: Parses beatmap notes to calculate the percentage of 4th, 8th, 12th, 16th, 24th, 32nd, and 64th notes.
- **Key Features**:
  - Prioritizes `.sm` files over `.ssc` for consistent parsing.
  - Automatically categorizes difficulty into **Easy**, **Medium**, **Hard**, or **Challenging** based on note density.
  - Outputs results to `rhythm_density_stats.csv`.

### 2. `categorize_tempos.py`
- **What it does**: Analyzes `#BPMS` and `#STOPS` tags to categorize songs into tempo buckets.
- **Categories**:
  - **Constant BPM**: Steady tempo throughout.
  - **Tempo change / BPM shift**: Discrete jumps in tempo.
  - **Gradual tempo change**: Smooth accelerando or ritardando.
  - **Rubato / Expressive**: Frequent timing changes or stops.
  - **Double-time / Half-time feel**: Sudden exact 2x or 0.5x BPM shifts.
- **Outputs**: `tempo_categories.json` and a statistical summary in the console.

### 3. `analyze_local_bpm.py`
- **What it does**: Extracts the exact beat positions and BPM values for every tempo change in a song.
- **Purpose**: Provides a "ground truth" dataset to verify if a model can predict *where* a tempo change occurs and *what* the local BPM is for a specific section.
- **Outputs**: `local_bpm_analysis.json`.

## File Placement

To use these tools, place your beatmap files in the following directory:
- **Input Path**: `src/musicForBeatmap/`
  - The scripts recursively scan all subdirectories within this folder.
  - They look for `.sm` and `.ssc` files.
  - If both an `.sm` and `.ssc` file exist for the same song, the `.sm` file is prioritized for analysis.

## Expected Outputs

After running the scripts, the following files will be generated in the **project root directory**:

1.  **`rhythm_density_stats.csv`**:
    - contains the quantization percentages (4th, 8th, etc.) and the calculated `Density_Label` for every difficulty level found.
2.  **`tempo_categories.json`**:
    - A JSON mapping of every song to its tempo category (Constant, Shift, etc.) and basic BPM stats.
3.  **`local_bpm_analysis.json`**:
    - A detailed breakdown of every tempo change event, including the beat position and the local BPM for that section.

## Usage

### Local or Direct Cluster Run
Since these scripts are lightweight, you can run them directly on a login node for up to ~500 songs:

```bash
python3 scripts/extract_rhythm_density.py
python3 scripts/categorize_tempos.py
python3 scripts/analyze_local_bpm.py
```

### Cluster Batch Run (SLURM)
If you have a very large dataset or prefer to use the cluster scheduler, use the provided batch script:

```bash
sbatch slurm_analyze_rhythm.sh
```

## Musical Logic
- **Quantization**: Notes are assigned to the widest possible quantization (e.g., a note on a 4th-beat position is always counted as a 4th note, even in a 16-line measure).
- **Density Labeling**:
  - **Challenging**: >5% notes are 24th+ OR >30% notes are 16th.
  - **Hard**: >10% notes are 16th OR contains any 24th+ notes.
  - **Easy**: >80% notes are 4th/8th.
  - **Medium**: Everything else.
