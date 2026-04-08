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


def plot_multi_tolerance_results(metrics_by_tol, target_sweeps, out_file, song_name):
    """
    metrics_by_tol: dict mapping tolerance (int) -> a dict mapping chunk_size (str) -> metrics dict
    e.g. {
       50:  { '20': {'f1_score':...}, '15': ... },
       100: { ... }
    }
    """
    # Sort chunks descending geometrically:
    chunks = sorted([int(k) for k in target_sweeps], reverse=True)
    labels = [f"{c}s Seg" for c in chunks]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Palette definition for tolerances
    colors = {
        50: 'tab:red',
        100: 'tab:blue',
        200: 'tab:green',
        300: 'tab:purple'
    }

    ax1.set_xlabel('Processing Audio Vector Size', fontweight='bold')
    ax1.set_ylabel('F1 Score (%)', fontweight='bold')

    # Line Plot for each Tolerance Bracket
    for tol, color in colors.items():
        if tol not in metrics_by_tol:
            continue
            
        f1_scores = [metrics_by_tol[tol][str(c)]['f1_score'] * 100 for c in chunks]
        
        ax1.plot(labels, f1_scores, color=color, marker='o', markersize=8, linewidth=2, label=f'±{tol}ms Tolerance')
        
        # Overlay the direct literal values securely over node
        for i, f1_val in enumerate(f1_scores):
            ax1.text(labels[i], f1_val + 0.8, f'{f1_val:.1f}%', ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

    ax1.set_ylim(0, 100) # F1 max
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower right', shadow=True)

    plt.title(f'LLM Context Recognition Scaled Against Multiple Tolerances\nSong: {song_name}', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    plt.savefig(out_file, dpi=300)
    print(f"\n📊 Successfully generated multi-line variance map to: {out_file}")


def main(gt_path, csv_dir):
    print("=" * 80)
    print(f"🎯 MULTI-TOLERANCE BASELINE: {os.path.basename(gt_path)}")
    print("=" * 80)
    
    try:
        true_onsets, source = load_ground_truth(gt_path)
    except Exception as e:
        print(f"❌ ground_truth error: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print(f"Total Source Onsets (Original): {len(true_onsets)}\n")

    target_sweeps = ["20", "15", "10", "5", "2"]
    tolerances_to_test = [50, 100, 200, 300]
    
    # Store matrix tracking all geometric thresholds
    matrix_metrics = {tol: {} for tol in tolerances_to_test}
    
    for size in target_sweeps:
        # Load local array sequentially
        pattern = os.path.join(csv_dir, f"qwen_onsets/*_Qwen_{size}s_*.csv")
        files = glob.glob(pattern)
        
        if not files:
            print(f"  ⚠️  [ {size}s ] Sweep Log Missing! Not found in Directory.")
            continue
            
        latest_file = max(files, key=os.path.getmtime)
        pred_onsets = []
        
        with open(latest_file, 'r') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row and row[0].strip():
                    try:
                        pred_onsets.append(int(float(row[0])))
                    except ValueError:
                        pass
        
        # Calculate for ALL tolerances using identical base logs explicitly 
        for tol in tolerances_to_test:
            metrics = evaluate_onsets(true_onsets, pred_onsets, tol)
            matrix_metrics[tol][size] = metrics

    # Debug sanity printout natively sequentially
    for tol in tolerances_to_test:
        print(f"\n--- Outputting Tolerance Limit: [ ±{tol}ms ] ---")
        for size in target_sweeps:
            if size in matrix_metrics[tol]:
                m = matrix_metrics[tol][size]
                f1_val = m['f1_score'] * 100
                print(f"  → [{size:>2}s Output] | Hits: {m['hits']:<4} | False Positives: {m['false_positives']:<4} | Missed: {m['false_negatives']:<4} | F1: {f1_val:.1f}%")

    # Output dynamic chart directly
    base_name = "Bad Ketchup"
    out_map = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis_reports", f"Qwen_Multi_Tolerance_Comparison_{base_name.replace(' ', '_')}.png")
    os.makedirs(os.path.dirname(out_map), exist_ok=True)
    
    plot_multi_tolerance_results(matrix_metrics, target_sweeps, out_map, base_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt",
        default="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/original_onsets_Bad_Ketchup_15032026143021.csv",
        help="Ground truth path"
    )
    parser.add_argument(
        "--csv_dir", 
        default="/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup",
        help="Target folder for logs"
    )
    args = parser.parse_args()
    
    # Tolerances are now fully automated securely inside main loop
    main(args.gt, args.csv_dir)
