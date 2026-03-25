"""
Evaluate Qwen onset predictions against MIDI-derived ground truth.

Reads:
  - Ground truth CSVs:  <dataset-dir>/onsets/midi_<song>_mp3_onsets.csv
  - Qwen predictions:   <dataset-dir>/predictions/<song>/Qwen_LoRA_onsets_*.txt

Usage:
    python3 scripts/evaluate_midi_onsets.py \\
        --dataset-dir /path/to/dataset \\
        --tolerance   50 \\
        --save-json   outputs/midi_eval_results.json
"""

import os
import sys
import re
import glob
import json
import argparse
import datetime
import numpy as np


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOLERANCE_MS  = 50.0


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_ground_truth_csv(csv_path: str) -> list[float]:
    """Load onset_ms column from midi_<song>_mp3_onsets.csv."""
    onsets = []
    with open(csv_path) as f:
        lines = f.readlines()
    for line in lines[1:]:  # skip header
        parts = line.strip().split(",")
        if len(parts) >= 2:
            try:
                onsets.append(float(parts[1]))
            except ValueError:
                pass
    return sorted(onsets)


def load_prediction_txt(txt_path: str) -> list[float]:
    """Load raw millisecond timestamps from a Qwen_LoRA_onsets_*.txt file."""
    onsets = []
    with open(txt_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                onsets.append(float(line))
            except ValueError:
                pass
    return sorted(onsets)


def find_latest_prediction(pred_dir: str, slug: str) -> str | None:
    """Find the latest Qwen_LoRA_onsets_*.txt for a given song slug."""
    pattern = os.path.join(pred_dir, slug, "Qwen_LoRA_onsets_*.txt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


# ── Scorer ────────────────────────────────────────────────────────────────────

def score_onsets(pred_ms: list[float], gt_ms: list[float],
                 tolerance_ms: float = TOLERANCE_MS) -> dict:
    if not pred_ms or not gt_ms:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0,
                "tp": 0, "fp": len(pred_ms), "fn": len(gt_ms),
                "n_predicted": len(pred_ms), "n_ground_truth": len(gt_ms)}

    pred_arr = np.array(pred_ms)
    gt_arr   = np.array(gt_ms)
    matched  = set()
    tp = 0

    for p in pred_arr:
        diffs    = np.abs(gt_arr - p)
        best_idx = int(np.argmin(diffs))
        if diffs[best_idx] <= tolerance_ms and best_idx not in matched:
            tp += 1
            matched.add(best_idx)

    fp = len(pred_arr) - tp
    fn = len(gt_arr)   - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision":       round(precision, 4),
        "recall":          round(recall, 4),
        "f1":              round(f1, 4),
        "tp":              tp,
        "fp":              fp,
        "fn":              fn,
        "n_predicted":     len(pred_arr),
        "n_ground_truth":  len(gt_arr),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate MIDI onset predictions")
    parser.add_argument("--dataset-dir", required=True,
                        help="Root of the prepared dataset (contains onsets/ and predictions/)")
    parser.add_argument("--tolerance", type=float, default=TOLERANCE_MS,
                        help=f"Match tolerance in ms (default: {TOLERANCE_MS})")
    parser.add_argument("--save-json", default=None,
                        help="Save full results as JSON at this path")
    args = parser.parse_args()

    onset_dir = os.path.join(args.dataset_dir, "onsets")
    pred_dir  = os.path.join(args.dataset_dir, "predictions")

    csv_files = sorted(glob.glob(os.path.join(onset_dir, "midi_*_mp3_onsets.csv")))
    if not csv_files:
        print(f"❌ No ground-truth CSVs found in: {onset_dir}")
        sys.exit(1)

    print(f"\n{'='*62}")
    print(f"  Qwen LoRA vs MIDI Ground Truth — F1 Evaluation")
    print(f"  Tolerance: ±{args.tolerance} ms  |  Songs: {len(csv_files)}")
    print(f"{'='*62}\n")

    all_results = {}
    total_tp = total_fp = total_fn = 0

    for csv_path in csv_files:
        # Derive slug from filename: midi_<slug>_mp3_onsets.csv
        base = os.path.basename(csv_path)
        slug = re.sub(r"^midi_", "", re.sub(r"_mp3_onsets\.csv$", "", base))
        display_name = slug.replace("_", " ")

        print(f"  🎵 {display_name}")
        gt_ms = load_ground_truth_csv(csv_path)
        print(f"     [GT] {len(gt_ms)} onsets loaded from {base}")

        pred_file = find_latest_prediction(pred_dir, slug)
        if not pred_file:
            print(f"     ⚠️  No prediction file found — skipping\n")
            continue
        pred_ms = load_prediction_txt(pred_file)
        print(f"     [Qwen] {len(pred_ms)} onsets loaded from {os.path.basename(pred_file)}")

        scores = score_onsets(pred_ms, gt_ms, args.tolerance)
        all_results[slug] = {**scores, "display_name": display_name}

        total_tp += scores["tp"]
        total_fp += scores["fp"]
        total_fn += scores["fn"]

        bar = "█" * int(scores["f1"] * 20)
        print(f"     TP:{scores['tp']}  FP:{scores['fp']}  FN:{scores['fn']}")
        print(f"     P:{scores['precision']:.1%}  R:{scores['recall']:.1%}  "
              f"F1:{scores['f1']:.1%}  [{bar:<20}]\n")

    # Micro-average
    if total_tp + total_fp > 0 and total_tp + total_fn > 0:
        micro_p  = total_tp / (total_tp + total_fp)
        micro_r  = total_tp / (total_tp + total_fn)
        micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)
                    if (micro_p + micro_r) > 0 else 0.0)

        print(f"{'='*62}")
        print(f"  OVERALL MICRO-AVERAGE ({len(all_results)} songs)")
        print(f"  Precision: {micro_p:.1%}  Recall: {micro_r:.1%}  F1: {micro_f1:.1%}")
        print(f"{'='*62}\n")

        if   micro_f1 >= 0.85: print("  ✅ EXCELLENT — F1 ≥ 85%")
        elif micro_f1 >= 0.70: print("  👍 GOOD      — F1 ≥ 70%")
        elif micro_f1 >= 0.50: print("  ⚠️  MODERATE  — F1 ≥ 50%")
        else:                  print("  ❌ POOR      — needs more data or epochs")

        if args.save_json:
            out = {
                "timestamp":    datetime.datetime.now().isoformat(),
                "tolerance_ms": args.tolerance,
                "songs":        all_results,
                "overall":      {"precision": round(micro_p, 4),
                                 "recall":    round(micro_r, 4),
                                 "f1":        round(micro_f1, 4)},
            }
            os.makedirs(os.path.dirname(args.save_json), exist_ok=True)
            with open(args.save_json, "w") as f:
                json.dump(out, f, indent=2)
            print(f"\n  📄 Results saved → {args.save_json}")


if __name__ == "__main__":
    main()
