"""
Visualize MIDI onset evaluation results from a JSON file.

Reads the output of evaluate_midi_onsets.py and generates a rich bar chart
showing Precision, Recall, and F1-Score for each song.

Usage:
    python3 scripts/visualize_midi_eval.py \\
        --results-json outputs/midi_eval_results.json \\
        --out-png      outputs/midi_eval_chart.png
"""

import os
import sys
import json
import argparse
import datetime


def main():
    parser = argparse.ArgumentParser(description="Visualize MIDI onset F1 results")
    parser.add_argument("--results-json", required=True,
                        help="Path to the JSON file from evaluate_midi_onsets.py")
    parser.add_argument("--out-png", default=None,
                        help="Output path for the chart PNG (default: same dir as JSON)")
    parser.add_argument("--tolerance", type=float, default=None,
                        help="Override tolerance label in the chart title")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed. Run:  pip install matplotlib")
        sys.exit(1)

    # ── Load JSON ─────────────────────────────────────────────────────────────
    with open(args.results_json) as f:
        data = json.load(f)

    songs   = list(data["songs"].keys())
    results = data["songs"]
    overall = data.get("overall", {})
    tolerance_ms = args.tolerance or data.get("tolerance_ms", 50)

    if not songs:
        print("❌ No song results found in JSON.")
        sys.exit(1)

    # ── Build arrays ──────────────────────────────────────────────────────────
    labels     = [results[s].get("display_name", s.replace("_", " ")) for s in songs]
    f1_scores  = [results[s]["f1"]        for s in songs]
    precisions = [results[s]["precision"] for s in songs]
    recalls    = [results[s]["recall"]    for s in songs]

    x     = np.arange(len(songs))
    width = 0.28

    # ── Color-code F1 bars ────────────────────────────────────────────────────
    def f1_color(f1):
        if   f1 >= 0.80: return "#2ecc71"   # green  — excellent
        elif f1 >= 0.60: return "#f39c12"   # orange — good
        elif f1 >= 0.40: return "#e67e22"   # amber  — moderate
        else:            return "#e74c3c"   # red    — poor

    bar_colors = [f1_color(f) for f in f1_scores]

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(14, len(songs) * 0.9), 7))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    bars_p = ax.bar(x - width, precisions, width, label="Precision",
                    color="#3498db", alpha=0.85, zorder=3)
    bars_r = ax.bar(x,          recalls,   width, label="Recall",
                    color="#9b59b6", alpha=0.85, zorder=3)
    bars_f = ax.bar(x + width,  f1_scores, width, label="F1-Score",
                    color=bar_colors,            zorder=3)

    # Overall average line
    avg_f1 = overall.get("f1", float(np.mean(f1_scores)))
    ax.axhline(avg_f1, color="#f1c40f", linewidth=1.8, linestyle="--",
               zorder=4, label=f"Avg F1 = {avg_f1:.1%}")

    # ── Labels on F1 bars ─────────────────────────────────────────────────────
    for bar, f1 in zip(bars_f, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.012,
                f"{f1:.0%}",
                ha="center", va="bottom",
                fontsize=8, color="white", fontweight="bold")

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9, color="white")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", color="white", fontsize=11)
    ax.set_xlabel("Song", color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#555")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(axis="y", color="#333", linestyle="--", linewidth=0.6, zorder=0)

    # ── Legend & title ────────────────────────────────────────────────────────
    excellent = mpatches.Patch(color="#2ecc71", label="F1 ≥ 80%  Excellent")
    good      = mpatches.Patch(color="#f39c12", label="F1 ≥ 60%  Good")
    moderate  = mpatches.Patch(color="#e67e22", label="F1 ≥ 40%  Moderate")
    poor      = mpatches.Patch(color="#e74c3c", label="F1 < 40%  Poor")

    legend1 = ax.legend(handles=[excellent, good, moderate, poor],
                        loc="upper right", fontsize=8,
                        facecolor="#0f3460", edgecolor="#555", labelcolor="white")
    ax.add_artist(legend1)
    ax.legend(loc="upper left", fontsize=9,
              facecolor="#0f3460", edgecolor="#555", labelcolor="white")

    ax.set_title(
        f"Qwen2-Audio LoRA — MIDI Onset Detection  (±{tolerance_ms:.0f} ms tolerance)\n"
        f"Overall:  P={overall.get('precision', 0):.1%}  "
        f"R={overall.get('recall', 0):.1%}  "
        f"F1={avg_f1:.1%}  |  {len(songs)} songs",
        color="white", fontsize=12, pad=14
    )

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.out_png is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out_png = os.path.join(
            os.path.dirname(args.results_json),
            f"midi_eval_chart_{ts}.png"
        )

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"✅ Chart saved → {args.out_png}")


if __name__ == "__main__":
    main()
