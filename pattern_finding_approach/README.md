# Data Science Topology Pattern Finder

This module processes standard `.ssc` & `.sm` rhythm game charts and applies a data science pipeline consisting of High-Dimensional Topology (UMAP & HDBSCAN) and Sequence Mining (Markov Chains & PrefixSpan) to discover standard patterns.

## Algorithm Overview

1. **Preprocessing**: Finds all `.ssc/.sm` files. Filters out non `dance-single` games and non 4-column charts. Binds all unique note characters (`0,1,2,M,L` etc.). Measures are dynamically expanded into a normalized $192 \times 4$ spatial matrix to give consistent dimensions for Machine Learning.
2. **Sequential Mining (PrefixSpan/Markov)**: Before collapsing the grid spatially, we compute standard Markov Chain transition probabilities (what step state follows what step state?) and PrefixSpan recurrent sub-sequences.
3. **The Compressor (PCA via One-Hot)**: The normalized matrices are One-Hot Encoded to represent characters cleanly without mathematical weight biases, and squashed down to 16 dimensions heavily dropping sparsity via Principal Component Analysis (PCA).
4. **The Grouper (HDBSCAN)**: This algorithm discovers dense high-dimensional spatial neighborhoods. Clusters represent highly repeating patterns (like repetitive staircases). Anything not standard/messy is labeled as `-1` (Noise).
5. **The Visualizer (UMAP)**: Drops the space into a 2D plot. Noise will appear as gray points while clusters will form distinct "islands" colored differently.

## Output Structure

Running the script creates a `results/` folder (or whatever you explicitly name via argument) containing:
- `difficulty_counts.csv`: All difficulties found and measure count.
- `character_distributions.csv`: Tally of every character (with the column identifying if it's new).
- `hdbscan_cluster_counts.csv`: Tally of the discovered patterns groups.
- `markov_chain_transitions.csv`: Sequential probability logic of notes.
- `prefixspan_frequent_patterns.csv`: Sub-pattern step strings over the measures.
- `measure_cluster_map.png`: The visual interactive output map.

## Requirements & HPC Installation

You need several python libraries installed either locally, or mapped in your `module load` python installation on the slurm cluster.

```bash
pip install pandas numpy scikit-learn umap-learn matplotlib prefixspan hdbscan
```

*(Note: Older scikit-learn doesn't carry HDBSCAN natively which is why we pip install it raw as well)*

## Running the Script

Run this from the terminal. 
By default, it targets `../../src/musicForBeatmap/`. You can overwrite the target path using `--target_dir`.

```bash
# Run on a local test drive (e.g. your local directory with 1 song)
python3 pattern_finding.py --target_dir "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup" --output_dir "test_results"

# Run it on the entire cluster root directory
python3 pattern_finding.py --target_dir "/path/to/massive/datasets/" --output_dir "massive_results"
```
