#!/usr/bin/env python3
import os
import glob
import json
import csv
import argparse
import matplotlib.pyplot as plt

# Try importing the F1 metrics logic native to the current repo
try:
    from compare_onsets import evaluate_onsets, load_ground_truth
except ImportError:
    print("❌ Critical: Please run this script directly within the 'scripts/' directory!")
    import sys; sys.exit(1)

def plot_sweep_results(metrics_dict, out_file, song_name, tolerance=50):
    """
    metrics_dict: { '20': { 'f1': 80.5, 'hits': 400, ... }, '15': ... }
    Generates a dual-axis Matplotlib Bar Chart.
    """
    chunks = sorted([int(k) for k in metrics_dict.keys()], reverse=True)
    f1_scores = [metrics_dict[str(c)]['f1_score'] * 100 for c in chunks]
    hits = [metrics_dict[str(c)]['hits'] for c in chunks]
    labels = [f"{c}s Segment" for c in chunks]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for F1 Scores
    color = 'tab:blue'
    ax1.set_xlabel('Processing Audio Vector Size')
    ax1.set_ylabel('F1 Score (%)', color=color, fontweight='bold')
    bars = ax1.bar(labels, f1_scores, color=color, alpha=0.7, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, max(f1_scores + [10]) * 1.2) # Give breathing room at the top

    # Add numeric labels to F1 Bars
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1.0, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold', color=color)

    # Line Plot for Raw Hits (True Positives)
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Exact Hits Detected (Count)', color=color2, fontweight='bold')
    line = ax2.plot(labels, hits, color=color2, marker='o', markersize=8, linewidth=2, linestyle='dashed')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, max(hits + [10]) * 1.2)

    for i, txt in enumerate(hits):
        ax2.annotate(f"{txt} hits", (labels[i], hits[i] + (max(hits)*0.05)), textcoords="offset points", xytext=(0,5), ha='center', color=color2)

    plt.title(f'LLM Onset Detection Stability by Temporal Bounds\nSong: {song_name} (Tolerance: ±{tolerance}ms)', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    plt.savefig(out_file, dpi=300)
    print(f"\n📊 Successfully rendered graphical map to: {out_file}")


def main(gt_path, csv_dir, tolerance):
    print("=" * 80)
    print(f"🎯 GROUND TRUTH BASELINE: {os.path.basename(gt_path)}")
    print("=" * 80)
    
    try:
        true_onsets, source = load_ground_truth(gt_path)
    except Exception as e:
        print(f"❌ ground_truth error: {e}")
        return
        
    print(f"Total Source Onsets (Original): {len(true_onsets)}\n")

    # The 5 sweeps we executed via the bash scripts
    target_sweeps = ["20", "15", "10", "5", "2"]
    sweep_results = {}
    
    for size in target_sweeps:
        # Search relative to out folder structure: .../Bad Ketchup/qwen_onsets/Bad_Ketchup_Qwen_{size}s_*.csv
        pattern = os.path.join(csv_dir, f"qwen_onsets/*_Qwen_{size}s_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            print(f"  ⚠️  [ {size}s ] Sweep Log Missing! Not found in Directory.")
            continue
            
        # Get the latest execution of this particular sweep run
        latest_file = max(files, key=os.path.getmtime)
        pred_onsets = []
        
        with open(latest_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)  # By-pass header 'onset_ms'
            for row in reader:
                if row and row[0].strip():
                    try:
                        pred_onsets.append(int(float(row[0])))
                    except ValueError:
                        pass
                        
        metrics = evaluate_onsets(true_onsets, pred_onsets, tolerance)
        sweep_results[size] = metrics
        
        f1_mod = metrics['f1_score'] * 100
        print(f"  → [{size:>2}s Output] | Hits: {metrics['hits']:<4} | False Positives: {metrics['false_positives']:<4} | Missed: {metrics['false_negatives']:<4} | F1: {f1_mod:.1f}%")

    if not sweep_results:
        print("\n❌ Critical: No sweep logs were returned. Ensure you rsync'd the output from HPC first.")
        return

    # Dump the visual map
    base_name = "Bad Ketchup" # Extracted dynamically normally, overriding strictly for this dataset demo
    out_map = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_reports", f"Qwen_Onset_Sweep_Comparison_{base_name.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(out_map), exist_ok=True)
    
    plot_sweep_results(sweep_results, out_map, base_name, tolerance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        default="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/original_onsets_Bad_Ketchup_15032026143021.csv",
        help="Ground truth: path to original beatmap CSV"
    )
    parser.add_argument(
        "--csv_dir", 
        default="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup",
        help="Target parent folder harboring the qwen_onsets logs"
    )
    parser.add_argument("--tolerance", type=int, default=50, help="Tolerance in ms")
    args = parser.parse_args()
    
    main(args.gt, args.csv_dir, args.tolerance)
