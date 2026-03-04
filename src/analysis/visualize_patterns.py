#!/usr/bin/env python3
"""
visualize_patterns.py

Graphical visualization of DDR-style beat-grid pattern analysis.

Single-task mode (default):
    python3 src/analysis/visualize_patterns.py --task task0001

Multi-task comparison mode (radar + grouped bar, Gemini only by default):
    python3 src/analysis/visualize_patterns.py --compare task0001 task0003 task0005 --model gemini
    python3 src/analysis/visualize_patterns.py --compare task0001 task0003 task0005 task0004 --model gemini --windows 4 8 16
"""

import sys
import argparse
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Project root ───────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from sort_and_analyze_beatmaps import (
    PATTERN_CLASSIFIERS, load_rows, sliding_window_patterns,
    BASE_DIR, REPORT_DIR, note_label
)

# ── Constants ─────────────────────────────────────────────────────────────────
ALL_PATTERNS = [name for name, _ in PATTERN_CLASSIFIERS] + ["Single Notes"]

PALETTE = {
    "Bracket":     "#ff9f43",
    "Spin":        "#f9ca24",
    "Candle":      "#badc58",
    "Gallop":      "#6ab04c",
    "Jack":        "#22a6b3",
    "Double Step": "#be2edd",
    "Jump":        "#eb4d4b",
    "Footswitch":  "#30336b",
    "Stream":      "#1e90ff",
    "Single Notes":"#95afc0",
}

# Distinct colors per task for comparison charts
TASK_COLORS = ["#e63946", "#2a9d8f", "#e9c46a", "#457b9d", "#f4a261", "#8338ec"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def collect_sorted_csvs(base_dir: Path, task_id: str, model_filter: str = "") -> list[Path]:
    """Return sorted CSV paths matching task_id and optional model keyword."""
    all_files = sorted(base_dir.rglob(f"*{task_id}*_sorted.csv"))
    if model_filter:
        all_files = [p for p in all_files if model_filter.lower() in p.name.lower()]
    return all_files


def build_avg_pct(csv_files: list[Path], window_rows: int) -> np.ndarray:
    """Compute average pattern % across all files for one window size.
    Returns array shape (n_patterns,).
    """
    if not csv_files:
        return np.zeros(len(ALL_PATTERNS))

    all_pct = []
    for p in csv_files:
        rows = load_rows(p)
        counts = dict(sliding_window_patterns(rows, window_rows).most_common())
        vec = np.array([counts.get(pat, 0) for pat in ALL_PATTERNS], dtype=float)
        total = vec.sum()
        if total > 0:
            all_pct.append(vec / total * 100.0)

    if not all_pct:
        return np.zeros(len(ALL_PATTERNS))
    return np.mean(all_pct, axis=0)


def build_matrix_from_rows(rows_dict: dict[str, list[dict]], window_rows: int):
    """Returns short_names[], raw_matrix (n_songs × n_patterns), pct_matrix (% per song)."""
    short_names, rows_all = [], []
    for label, rows in rows_dict.items():
        counts = dict(sliding_window_patterns(rows, window_rows).most_common())
        rows_all.append(counts)
        # clean label for chart
        short_names.append(label.replace(" (", "\n("))

    n, m = len(rows_dict), len(ALL_PATTERNS)
    raw = np.zeros((n, m))
    for i, counts in enumerate(rows_all):
        for j, pat in enumerate(ALL_PATTERNS):
            raw[i, j] = counts.get(pat, 0)

    totals = raw.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    pct = raw / totals * 100.0
    return short_names, raw, pct

def build_matrix(csv_files: list[Path], window_rows: int):
    """Legacy wrapper for building from CSV files."""
    rows_dict = {}
    for p in csv_files:
        rows = load_rows(p)
        parts = p.stem.replace("_sorted", "").split("_")
        song  = parts[0][:16]
        model = parts[2][:10] if len(parts) > 2 else ""
        label = f"{song} ({model})"
        rows_dict[label] = rows
    return build_matrix_from_rows(rows_dict, window_rows)


# ── Single-task charts ────────────────────────────────────────────────────────

def plot_stacked_bar(short_names, pct, window_rows, out_path):
    n = len(short_names)
    fig, ax = plt.subplots(figsize=(15, max(5, n * 0.6)))
    lefts = np.zeros(n)

    for j, pat in enumerate(ALL_PATTERNS):
        vals = pct[:, j]
        ax.barh(range(n), vals, left=lefts, color=PALETTE.get(pat, "#888"),
                label=pat, edgecolor="white", linewidth=0.3)
        for i, (v, l) in enumerate(zip(vals, lefts)):
            if v > 7:
                ax.text(l + v / 2, i, f"{v:.0f}%",
                        va="center", ha="center", fontsize=7.5,
                        color="white", fontweight="bold")
        lefts += vals

    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel("% of windows classified as each pattern", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_title(f"Pattern Mix per Song  —  {note_label(window_rows)}",
                 fontsize=13, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=7.5, ncol=2, framealpha=0.85)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Stacked bar  → {out_path}")


def plot_heatmap(short_names, raw, pct, window_rows, out_path):
    active = np.where(raw.sum(axis=0) > 0)[0]
    if not len(active):
        return
    sub_raw = raw[:, active]
    sub_pct = pct[:, active]
    pats = [ALL_PATTERNS[j] for j in active]
    n = len(short_names)

    fig, ax = plt.subplots(figsize=(max(9, len(pats) * 1.1), max(5, n * 0.5)))
    im = ax.imshow(np.log1p(sub_raw), cmap="YlOrRd", aspect="auto", origin="upper")

    ax.set_xticks(range(len(pats)))
    ax.set_xticklabels(pats, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(short_names, fontsize=7)

    for i in range(n):
        for j in range(len(pats)):
            v_raw = int(sub_raw[i, j])
            v_pct = sub_pct[i, j]
            if v_raw > 0:
                dark = np.log1p(sub_raw[i, j]) > 3
                ax.text(j, i, f"{v_raw}\n{v_pct:.0f}%",
                        ha="center", va="center", fontsize=6.5,
                        color="white" if dark else "black")

    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02).set_label("log(1 + count)", fontsize=9)
    ax.set_title(f"Pattern Heatmap  —  {note_label(window_rows)}\nCells: raw count / % of windows",
                 fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  ✓ Heatmap      → {out_path}")


# ── Multi-task comparison charts ──────────────────────────────────────────────

# Colors for note-grid window sizes (one per grid level)
GRID_COLORS = {
    1:  "#e63946",   # Whole note  — red
    2:  "#f4a261",   # Half note   — orange
    4:  "#2a9d8f",   # Quarter     — teal
    8:  "#457b9d",   # Eighth      — blue
    12: "#8338ec",   # Triplet     — purple
    16: "#e9c46a",   # Sixteenth   — yellow
}


def _active_from_arrays(arrays: list[np.ndarray]) -> list[str]:
    """Always return ALL patterns as radar axes so none are silently hidden.
    Patterns with 0% will collapse to the centre of the spider, clearly showing
    which patterns the model never generates.
    """
    return ALL_PATTERNS        # show every pattern, no filtering


def plot_all_windows_radar(avg_dict_per_window: dict,
                           window_rows_list: list[int],
                           task_labels: list[str],
                           combined_label: str,
                           out_path: str,
                           chart_title: str = "Beat Pattern Profile — Average"):
    """
    ONE radar chart showing ALL note-grid window sizes as separate polygons.
    Each polygon = average across ALL tasks and ALL songs for that window size.

    avg_dict_per_window: {window_rows: {task_id: avg_array(n_patterns)}}
    """
    # Collapse tasks → mean per window size
    win_avgs = {}   # {window_rows: avg_array across all tasks}
    for w in window_rows_list:
        task_arrays = list(avg_dict_per_window[w].values())
        if task_arrays:
            win_avgs[w] = np.mean(np.stack(task_arrays, axis=0), axis=0)

    if not win_avgs:
        print("  ⚠  No data for combined window radar.")
        return

    # Active patterns across all windows
    active_pats = _active_from_arrays(list(win_avgs.values()))
    active_idx  = [ALL_PATTERNS.index(p) for p in active_pats]

    if len(active_pats) < 3:
        print("  ⚠  Not enough active patterns for radar.")
        return

    N = len(active_pats)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(polar=True))

    # ── Square-root radial transform ──────────────────────────────────────────
    # sqrt(val) stretches the 0-20 % zone visually so small differences are
    # easy to see; ring labels show the ACTUAL % so nothing is misleading.
    def r(v): return np.sqrt(np.maximum(v, 0))   # transform value → radius

    for w, avg in win_avgs.items():
        raw_vals = avg[active_idx].tolist() + [avg[active_idx[0]]]
        t_vals   = [r(v) for v in raw_vals]        # transformed radii
        color    = GRID_COLORS.get(w, "#888888")
        label    = note_label(w)
        ax.plot(angles, t_vals, color=color, linewidth=2.8, label=label)
        ax.fill(angles, t_vals, color=color, alpha=0.08)
        # Annotate peak with REAL % value
        mx_raw = max(raw_vals[:-1])
        if mx_raw > 1:
            mi = raw_vals.index(mx_raw)
            ax.annotate(f"{mx_raw:.0f}%",
                        xy=(angles[mi], r(mx_raw)),
                        xytext=(angles[mi], r(mx_raw) + 0.6),
                        fontsize=8.5, color=color, fontweight="bold",
                        ha="center")

    # ── Custom rings labeled with real % ──────────────────────────────────────
    ring_pcts  = [1, 2, 3, 4, 5, 10, 20, 40, 60]
    ring_radii = [r(p) for p in ring_pcts]
    ax.set_yticks(ring_radii)
    ax.set_yticklabels([f"{p}%" for p in ring_pcts], fontsize=8, color="#555")
    ax.set_ylim(0, r(70))

    # ── Colour spoke labels: red = absent everywhere, black = detected ─────────
    # Compute total avg across all windows for each active pattern
    all_win_avgs = np.stack(list(win_avgs.values()), axis=0)  # (n_windows, n_patterns)
    pat_total    = all_win_avgs[:, active_idx].sum(axis=0)    # sum across windows
    spoke_colors = ["#cc0000" if t < 0.001 else "#111111" for t in pat_total]

    ax.set_xticks(angles[:-1])
    tick_labels = ax.set_xticklabels(active_pats, fontsize=11, fontweight="bold")
    for lbl, color in zip(tick_labels, spoke_colors):
        lbl.set_color(color)

    ax.set_rlabel_position(15)
    n_tasks   = len(list(list(avg_dict_per_window.values())[0].keys())) if avg_dict_per_window else 0
    ax.set_title(f"{chart_title}\n"
                 "Black label = detected at least once   |   Red label = never detected",
                 fontsize=13, fontweight="bold", pad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.42, 1.18), fontsize=11,
              title="Note Grid", title_fontsize=10, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Combined radar → {out_path}")


def plot_comparison_grouped(avg_dict: dict, window_rows_list: list[int],
                            task_labels: list[str], out_path: str):
    """
    Grouped bar: X = patterns, groups = tasks, bars within group = window sizes.
    Shows how each TASK's pattern mix looks at each window size.
    """
    task_ids = list(avg_dict[window_rows_list[0]].keys())

    # Build combined: for each task, avg pct per pattern per window (mean across windows)
    # Actually: show one bar per (task × window) combination
    # Pattern on X, bars grouped by task, coloured by window size
    avgs_by_task_window = {}   # {(task_id, window_rows): avg_array}
    for w in window_rows_list:
        for task_id, avgs in avg_dict[w].items():
            avgs_by_task_window[(task_id, w)] = avgs

    # Find active patterns
    all_avgs = np.stack(list(avgs_by_task_window.values()), axis=0)
    active = np.where(all_avgs.max(axis=0) > 1.0)[0]
    pats = [ALL_PATTERNS[j] for j in active]

    if not pats:
        print("  ⚠  No patterns above 1% for grouped bar.")
        return

    n_tasks   = len(task_ids)
    n_windows = len(window_rows_list)
    # Each task gets a group, within group bars = window sizes
    group_width = 0.8
    bar_width   = group_width / n_windows
    x = np.arange(len(pats))

    # Window-size colour map
    win_cmap = plt.cm.plasma
    win_colors = [win_cmap(i / max(n_windows - 1, 1)) for i in range(n_windows)]

    fig, axes = plt.subplots(1, n_tasks, figsize=(max(12, len(pats) * n_tasks * 0.55), 7),
                              sharey=True)
    if n_tasks == 1:
        axes = [axes]

    for ti, task_id in enumerate(task_ids):
        ax = axes[ti]
        label = task_labels[ti] if ti < len(task_labels) else task_id
        for wi, w in enumerate(window_rows_list):
            avgs = avgs_by_task_window[(task_id, w)]
            vals = avgs[active]
            offset = (wi - n_windows / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, vals, bar_width * 0.9,
                          color=win_colors[wi], edgecolor="white", linewidth=0.4,
                          label=note_label(w))
            for bar, v in zip(bars, vals):
                if v > 3:
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                            f"{v:.0f}%", ha="center", va="bottom", fontsize=7,
                            color="#333")

        ax.set_xticks(x)
        ax.set_xticklabels(pats, rotation=30, ha="right", fontsize=9)
        ax.set_title(label, fontsize=11, fontweight="bold", color=TASK_COLORS[ti % len(TASK_COLORS)])
        ax.set_ylabel("Avg % of windows" if ti == 0 else "", fontsize=10)
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        if ti == 0:
            ax.legend(fontsize=8, framealpha=0.85, title="Window size")

    fig.suptitle("Task Comparison — Pattern Mix by Window Size\n(avg % across all Gemini songs)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Grouped bars → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise beat-grid pattern analysis.")
    parser.add_argument("--task",    default="task0001",
                        help="Single task ID for per-song charts (default: task0001)")
    parser.add_argument("--compare", nargs="+", default=None,
                        help="Multiple task IDs to compare on radar + grouped bar, "
                             "e.g. --compare task0001 task0003 task0005")
    parser.add_argument("--model",   default="gemini",
                        help="Model keyword filter for --compare mode (default: gemini, "
                             "use '' for all models)")
    
    # ── Original SSC mode ─────────────────────────────────────────────────
    parser.add_argument("--original",   action="store_true",
                        help="Analyse original .ssc beatmap files instead of AI CSVs")
    parser.add_argument("--difficulty", default=None,
                        help="(--original only) Filter to one difficulty, e.g. Hard")
    parser.add_argument("--song",       default=None,
                        help="(--original only) Filter to one specific song folder name, e.g. 'Bad Ketchup'")
    
    # ── Shared ────────────────────────────────────────────────────────────
    parser.add_argument("--windows", nargs="+", type=int, default=[2, 4, 8, 16],
                        help="Beat-grid window sizes (default: 4 8 16)")
    parser.add_argument("--no-sort", action="store_true")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── COMPARE MODE ─────────────────────────────────────────────────────────
    if args.compare:
        task_ids = args.compare
        model_kw = args.model
        print(f"\n{'='*62}")
        print(f"  MULTI-TASK COMPARISON  model_filter='{model_kw}'")
        print(f"  Tasks  : {task_ids}")
        print(f"  Windows: {[note_label(w) for w in args.windows]}")
        print(f"{'='*62}\n")

        # Build avg_pct per (window, task)
        # avg_dict_per_window: {window_rows: {task_id: avg_array}}
        avg_dict_per_window = {w: {} for w in args.windows}
        task_labels = []

        for task_id in task_ids:
            csv_files = collect_sorted_csvs(BASE_DIR, task_id, model_kw)
            if not csv_files:
                print(f"  ⚠  No {'Gemini ' if model_kw else ''}files for {task_id} — skipping")
                continue
            print(f"  {task_id}: {len(csv_files)} files")
            # Friendly label: task ID + count
            task_labels.append(f"{task_id} (n={len(csv_files)})")
            for w in args.windows:
                avg_dict_per_window[w][task_id] = build_avg_pct(csv_files, w)

        # Only keep tasks that were found
        found_tasks = list(avg_dict_per_window[args.windows[0]].keys())
        if not found_tasks:
            print("❌  No files found for any task. Check --model filter.")
            return

        suffix = "_".join(found_tasks) + f"_{model_kw}"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        print("\nGenerating comparison charts…")

        # ONE combined radar — all note grids as polygons on a single chart
        task_ids_str = " ".join(found_tasks)
        plot_all_windows_radar(
            avg_dict_per_window, args.windows, task_labels,
            task_ids_str,
            out_dir / f"compare_radar_all_grids_{suffix}_{ts}.png",
            chart_title="Beat Pattern Profile — Average for Gemini Model"
        )

        # Grouped bar — side-by-side tasks × window sizes
        plot_comparison_grouped(
            avg_dict_per_window, args.windows, task_labels,
            out_dir / f"compare_grouped_{suffix}_{ts}.png"
        )

    # ── ORIGINAL BEATMAP MODE ─────────────────────────────────────────────────
    elif args.original:
        from sort_and_analyze_beatmaps import load_all_original_beatmaps
        
        diff_label = args.difficulty or "all difficulties"
        song_label = f"\"{args.song}\"" if args.song else "all songs"
        print(f"\n{'='*62}")
        print(f"  ORIGINAL BEATMAP MODE  (difficulty: {diff_label}, song: {song_label})")
        print(f"{'='*62}\n")
        
        named_rows = load_all_original_beatmaps(BASE_DIR, args.difficulty, args.song)
        if not named_rows:
            print("❌  No SSC charts found.")
            return
        
        # Build clean source label for filenames
        suffix = "original"
        if args.song:
            # strip spaces from filename
            suffix += f"-{args.song.replace(' ', '')}"
        suffix += f"-{args.difficulty}" if args.difficulty else "-all"
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        all_data = []
        for w in args.windows:
            print(f"  Building matrix for {note_label(w)}…")
            names, raw, pct = build_matrix_from_rows(named_rows, w)
            all_data.append((w, names, raw, pct))

        print("\nGenerating charts…")
        for w, names, raw, pct in all_data:
            plot_stacked_bar(names, pct, w,
                             out_dir / f"pct_stacked_{suffix}_{note_label(w).split()[0]}_{ts}.png")

        # One combined radar — all window sizes as polygons on single chart
        avg_by_window = {}
        # compute means
        for w, names, raw, pct in all_data:
            avg_by_window[w] = {suffix: pct.mean(axis=0)}

        plot_all_windows_radar(
            avg_by_window, args.windows, [suffix], suffix,
            out_dir / f"pct_radar_{suffix}_all_grids_{ts}.png",
            chart_title="Beat Pattern Profile — Official Human Charts (Fraxtil)"
        )

        for w, names, raw, pct in all_data:
            plot_heatmap(names, raw, pct, w,
                         out_dir / f"pct_heatmap_{suffix}_{note_label(w).split()[0]}_{ts}.png")

    # ── SINGLE-TASK MODE ──────────────────────────────────────────────────────
    else:
        task_id = args.task
        if not args.no_sort:
            from sort_and_analyze_beatmaps import stage_sort
            stage_sort(BASE_DIR, task_id)

        csv_files = collect_sorted_csvs(BASE_DIR, task_id)
        if not csv_files:
            print(f"❌  No sorted CSVs found for task '{task_id}'")
            sys.exit(1)

        suffix = task_id
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*60}")
        print(f"  PATTERN VISUALIZATION  task={task_id}  |  {len(csv_files)} charts")
        print(f"{'='*60}\n")

        all_data = []
        for w in args.windows:
            print(f"  Building matrix for {note_label(w)}…")
            names, raw, pct = build_matrix(csv_files, w)
            all_data.append((w, names, raw, pct))

        print("\nGenerating charts…")
        for w, names, raw, pct in all_data:
            plot_stacked_bar(names, pct, w,
                             out_dir / f"pct_stacked_{suffix}_{note_label(w).split()[0]}_{ts}.png")

        # One combined radar — all window sizes as polygons on single chart
        avg_by_window = {}
        for w, names, raw, pct in all_data:
            avg_by_window[w] = {task_id: pct.mean(axis=0)}

        plot_all_windows_radar(
            avg_by_window, args.windows, [task_id], task_id,
            out_dir / f"pct_radar_{suffix}_all_grids_{ts}.png",
            chart_title=f"Beat Pattern Profile — Gemini ({task_id})"
        )

        for w, names, raw, pct in all_data:
            plot_heatmap(names, raw, pct, w,
                         out_dir / f"pct_heatmap_{suffix}_{note_label(w).split()[0]}_{ts}.png")


    print(f"\n✅  All charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
