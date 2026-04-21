"""
extract_tempo_changes.py
========================
Extracts structured tempo change timelines from songs in the musicForBeatmap
dataset. For each song it produces:

  1. Ground Truth (from .sm/.ssc):  list of (beat, bpm) pairs exactly as a
     human composer wrote them.
  2. Librosa Detection:             list of (time_sec, bpm) pairs detected by
     sliding-window beat tracking over the audio.
  3. Comparison CSV:                Per-song comparison of GT vs Librosa, with
     a drift error metric for each detected section.

Output per song row (training-ready):
  Song_Name | Total_Duration_s | Num_GT_BPM_Changes | GT_BPM_Timeline (JSON) |
  Num_Librosa_Changes | Librosa_BPM_Timeline (JSON) | Mean_BPM_Error | Is_Variable_Tempo

Usage:
    python3 onsetdetection/extract_tempo_changes.py \\
        --batch_dir src/musicForBeatmap \\
        --window_size 20 \\
        --out_csv onsetdetection/Tempo_Change_Analysis.csv

    # For a single song folder:
    python3 onsetdetection/extract_tempo_changes.py \\
        --song_dir "src/musicForBeatmap/Fraxtil/Springtime"
"""

import os
import re
import sys
import csv
import json
import math
import argparse
import numpy as np
import librosa

# ── Constants ──────────────────────────────────────────────────────────────────

ROOT = "/data/mg546924/llm_beatmap_generator"
SEP  = "─" * 72

# ── SM/SSC Ground Truth Parser ─────────────────────────────────────────────────

def parse_sm_bpm_changes(sm_path: str) -> list[dict]:
    """
    Reads a .sm or .ssc file and returns ALL BPM change points.

    Returns a list of dicts:
      [{"beat": 0.0,   "bpm": 181.68},
       {"beat": 304.0, "bpm": 90.84},
       {"beat": 311.0, "bpm": 181.68}]

    For constant-BPM songs this will have exactly 1 entry.
    """
    if not sm_path or not os.path.exists(sm_path):
        return []

    try:
        with open(sm_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        # Grab the FIRST #BPMS: block (global header, not per-chart)
        m = re.search(r"#BPMS:(.*?);", content, re.DOTALL)
        if not m:
            return []

        bpms_str = m.group(1).strip()
        changes = []
        for entry in bpms_str.split(","):
            entry = entry.strip()
            if "=" not in entry:
                continue
            beat_str, bpm_str = entry.split("=", 1)
            try:
                beat = float(beat_str.strip())
                bpm  = round(float(bpm_str.strip()), 3)
                changes.append({"beat": beat, "bpm": bpm})
            except ValueError:
                continue

        return sorted(changes, key=lambda x: x["beat"])

    except Exception as e:
        print(f"  ⚠  Could not parse BPM from {sm_path}: {e}")
        return []


def parse_sm_offset(sm_path: str) -> float:
    """Returns #OFFSET value in seconds (0.0 if not found)."""
    try:
        with open(sm_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        m = re.search(r"#OFFSET:([-0-9.]+);", content)
        return float(m.group(1)) if m else 0.0
    except Exception:
        return 0.0


def bpm_changes_to_time_timeline(changes: list[dict], offset: float = 0.0) -> list[dict]:
    """
    Converts beat-based BPM change list → time-based timeline.

    Input:  [{"beat": 0, "bpm": 181.68}, {"beat": 304, "bpm": 90.84}, ...]
    Output: [{"time_sec": -0.07, "beat": 0,   "bpm": 181.68, "section_len_s": ...},
             {"time_sec": 100.4, "beat": 304, "bpm": 90.84,  "section_len_s": ...}, ...]
    """
    if not changes:
        return []

    timeline = []
    cumulative_time = offset  # starts from the audio offset

    for i, ch in enumerate(changes):
        entry = {
            "beat":     ch["beat"],
            "bpm":      ch["bpm"],
            "time_sec": round(cumulative_time, 4),
        }

        # Compute duration of this segment
        if i + 1 < len(changes):
            beat_delta = changes[i + 1]["beat"] - ch["beat"]
            seg_len    = beat_delta * (60.0 / ch["bpm"])
        else:
            seg_len    = None  # last segment goes to end of song

        entry["section_len_s"] = round(seg_len, 3) if seg_len is not None else None
        timeline.append(entry)

        if seg_len is not None:
            cumulative_time += seg_len

    return timeline


# ── Librosa Tempo Detection ────────────────────────────────────────────────────

def detect_tempo_changes_librosa(
    audio_path: str,
    window_size_s: float = 20.0,
    hop_size_s:    float = 10.0,
    min_bpm_delta: float = 3.0,
) -> tuple[list[dict], float]:
    """
    Slides a window over the audio and estimates BPM per window.
    Merges consecutive windows with similar BPM into a single segment.

    Args:
        audio_path:    Path to .ogg/.mp3/.wav
        window_size_s: Size of each analysis window in seconds (default 20s)
        hop_size_s:    Hop between windows in seconds (default 10s)
        min_bpm_delta: Minimum BPM difference to count as a tempo change (default 3)

    Returns:
        (timeline, total_duration_s)
        timeline: [{"time_sec": 5.0, "bpm": 181.0, "section_len_s": 60.0}, ...]
    """
    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        print(f"  ⚠  Could not load audio {audio_path}: {e}")
        return [], 0.0

    total_duration = librosa.get_duration(y=y, sr=sr)
    window_samples = int(window_size_s * sr)
    hop_samples    = int(hop_size_s * sr)

    raw_segments = []  # [(time_sec, bpm)]

    for start in range(0, len(y), hop_samples):
        segment = y[start : start + window_samples]

        # Need at least 4 seconds for a meaningful estimate
        if len(segment) < sr * 4:
            break

        try:
            tempo, _ = librosa.beat.beat_track(y=segment, sr=sr)
            bpm = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
            bpm = round(bpm, 2)
        except Exception:
            continue

        start_sec = round(start / sr, 3)
        raw_segments.append((start_sec, bpm))

    if not raw_segments:
        return [], total_duration

    # ── Merge consecutive windows with similar BPM ──
    merged = []
    seg_start, seg_bpm = raw_segments[0]
    seg_bpms = [seg_bpm]

    for t, bpm in raw_segments[1:]:
        if abs(bpm - np.mean(seg_bpms)) <= min_bpm_delta:
            seg_bpms.append(bpm)
        else:
            merged.append({
                "time_sec": seg_start,
                "bpm":      round(float(np.median(seg_bpms)), 2),
            })
            seg_start = t
            seg_bpms  = [bpm]

    # Final segment
    merged.append({
        "time_sec": seg_start,
        "bpm":      round(float(np.median(seg_bpms)), 2),
    })

    # Add section lengths
    for i, seg in enumerate(merged):
        if i + 1 < len(merged):
            seg["section_len_s"] = round(merged[i + 1]["time_sec"] - seg["time_sec"], 2)
        else:
            seg["section_len_s"] = round(total_duration - seg["time_sec"], 2)

    return merged, round(total_duration, 2)


# ── Comparison Metric ──────────────────────────────────────────────────────────

def compare_timelines(gt_timeline: list[dict], lib_timeline: list[dict]) -> dict:
    """
    For each Librosa-detected segment, find the closest GT segment in time
    and compute the BPM error.

    Returns:
      {"per_segment_errors": [...], "mean_bpm_error": float, "max_bpm_error": float}
    """
    if not gt_timeline or not lib_timeline:
        return {"per_segment_errors": [], "mean_bpm_error": None, "max_bpm_error": None}

    errors = []
    for lib_seg in lib_timeline:
        lib_t   = lib_seg["time_sec"]
        lib_bpm = lib_seg["bpm"]

        # Find the GT segment that was active at lib_t
        active_gt = gt_timeline[0]
        for gt_seg in gt_timeline:
            if gt_seg["time_sec"] <= lib_t:
                active_gt = gt_seg
            else:
                break

        gt_bpm = active_gt["bpm"]
        err    = abs(lib_bpm - gt_bpm)
        pct    = round(err / gt_bpm * 100, 2)

        errors.append({
            "lib_time_sec": lib_t,
            "lib_bpm":      lib_bpm,
            "gt_bpm":       gt_bpm,
            "abs_error":    round(err, 2),
            "pct_error":    pct,
        })

    mean_err = round(float(np.mean([e["abs_error"] for e in errors])), 3) if errors else None
    max_err  = round(float(np.max ([e["abs_error"] for e in errors])), 3) if errors else None

    return {
        "per_segment_errors": errors,
        "mean_bpm_error":     mean_err,
        "max_bpm_error":      max_err,
    }


# ── File Finders ───────────────────────────────────────────────────────────────

def find_audio_file(folder: str) -> str | None:
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".ogg", ".mp3", ".wav")) and not f.startswith("._"):
            return os.path.join(folder, f)
    return None


def find_sm_file(folder: str) -> str | None:
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".ssc", ".sm")) and not f.startswith("._"):
            return os.path.join(folder, f)
    return None


# ── Single Song Analysis ───────────────────────────────────────────────────────

def analyze_song(song_dir: str, window_size_s: float = 20.0) -> dict | None:
    """
    Full analysis for one song folder. Returns a summary dict or None if skipped.
    """
    song_name  = os.path.basename(song_dir)
    audio_path = find_audio_file(song_dir)
    sm_path    = find_sm_file(song_dir)

    if not audio_path:
        return None

    print(f"\n🎵 {song_name}")
    print(f"   Audio : {os.path.basename(audio_path)}")
    print(f"   SM    : {os.path.basename(sm_path) if sm_path else 'NONE'}")

    # ── Ground Truth ──
    gt_changes = parse_sm_bpm_changes(sm_path)
    offset      = parse_sm_offset(sm_path)
    gt_timeline = bpm_changes_to_time_timeline(gt_changes, offset)

    if gt_timeline:
        print(f"   GT BPM changes: {len(gt_timeline)}")
        for seg in gt_timeline:
            dur = f"{seg['section_len_s']:.1f}s" if seg["section_len_s"] else "until end"
            print(f"     beat {seg['beat']:>8.2f}  @  t={seg['time_sec']:>8.3f}s  →  {seg['bpm']} BPM  [{dur}]")
    else:
        print("   ⚠  No GT BPM data found in SM/SSC")

    # ── Librosa Detection ──
    print(f"   Detecting tempo with Librosa (window={window_size_s}s)...")
    lib_timeline, total_dur = detect_tempo_changes_librosa(audio_path, window_size_s=window_size_s)

    if lib_timeline:
        print(f"   Librosa detected {len(lib_timeline)} tempo segment(s):")
        for seg in lib_timeline:
            dur = f"{seg['section_len_s']:.1f}s"
            print(f"     t={seg['time_sec']:>8.3f}s  →  {seg['bpm']} BPM  [{dur}]")
    else:
        print("   ⚠  Librosa detected no tempo segments")

    # ── Comparison ──
    comparison  = compare_timelines(gt_timeline, lib_timeline)
    mean_err    = comparison["mean_bpm_error"]
    max_err     = comparison["max_bpm_error"]
    is_variable = len(gt_changes) > 1

    if mean_err is not None:
        print(f"   📊 Mean BPM error vs GT: {mean_err:.2f}  |  Max: {max_err:.2f}")
    if is_variable:
        print(f"   ⚡ VARIABLE TEMPO SONG  ({len(gt_changes)} BPM zones)")

    return {
        "Song_Name":             song_name,
        "Audio_File":            os.path.basename(audio_path),
        "SM_File":               os.path.basename(sm_path) if sm_path else "",
        "Total_Duration_s":      total_dur,
        "Is_Variable_Tempo":     is_variable,
        "Num_GT_BPM_Changes":    len(gt_changes),
        "GT_BPM_Timeline":       json.dumps(gt_timeline),
        "Num_Librosa_Segments":  len(lib_timeline),
        "Librosa_BPM_Timeline":  json.dumps(lib_timeline),
        "Mean_BPM_Error":        mean_err,
        "Max_BPM_Error":         max_err,
        "Per_Segment_Errors":    json.dumps(comparison["per_segment_errors"]),
    }


# ── Batch Runner ───────────────────────────────────────────────────────────────

def run_batch(batch_dir: str, out_csv: str, window_size_s: float = 20.0):
    print(f"\n{SEP}")
    print(f"🔍  TEMPO CHANGE EXTRACTION — {batch_dir}")
    print(f"    Window size : {window_size_s}s")
    print(f"    Output CSV  : {out_csv}")
    print(SEP + "\n")

    # Collect song dirs (same spider logic as verify_model_bpm)
    song_dirs = []
    for root_dir, _, files in os.walk(batch_dir, followlinks=True):
        if os.path.basename(root_dir).startswith("_"):
            continue
        if any(f.lower().endswith((".ogg", ".mp3", ".wav")) and not f.startswith("._") for f in files):
            song_dirs.append(root_dir)

    song_dirs = sorted(set(song_dirs))
    print(f"Found {len(song_dirs)} song directories.\n")

    rows     = []
    variable = 0
    constant = 0
    errors   = []

    for idx, song_dir in enumerate(song_dirs):
        print(f"[{idx+1}/{len(song_dirs)}]", end=" ", flush=True)
        try:
            result = analyze_song(song_dir, window_size_s=window_size_s)
            if result:
                rows.append(result)
                if result["Is_Variable_Tempo"]:
                    variable += 1
                else:
                    constant += 1
        except Exception as e:
            import traceback
            name = os.path.basename(song_dir)
            print(f"\n  ❌ ERROR on {name}: {e}")
            traceback.print_exc()
            errors.append(name)

    # ── Write CSV ──
    if rows:
        fieldnames = list(rows[0].keys())
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"\n{SEP}")
    print(f"✅  DONE")
    print(f"   Songs processed  : {len(rows)}")
    print(f"   Constant BPM     : {constant}")
    print(f"   Variable BPM     : {variable}  ← training-rich!")
    print(f"   Errors skipped   : {len(errors)}")
    print(f"   Output           : {out_csv}")
    print(SEP + "\n")

    if variable > 0:
        print("🔥 Variable BPM songs are particularly valuable for training —")
        print("   the model needs to learn to handle tempo changes mid-song.\n")


# ── Quick Single-Song Debug Mode ───────────────────────────────────────────────

def run_single(song_dir: str, window_size_s: float = 20.0):
    print(f"\n{SEP}")
    print(f"🔬  SINGLE SONG ANALYSIS")
    print(SEP)
    result = analyze_song(song_dir, window_size_s=window_size_s)
    if result:
        print(f"\n{'─'*72}")
        print("📋  FULL JSON RESULT:")
        print(json.dumps(result, indent=2))
    else:
        print("❌  No audio found in that directory.")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract and compare tempo change timelines from a beatmap dataset"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--batch_dir",  help="Root of the music pack (recursive)")
    group.add_argument("--song_dir",   help="Single song folder for quick debug")

    parser.add_argument(
        "--window_size", type=float, default=20.0,
        help="Librosa sliding window size in seconds (default: 20)"
    )
    parser.add_argument(
        "--out_csv",
        default=os.path.join(ROOT, "onsetdetection", "Tempo_Change_Analysis.csv"),
        help="Where to save the output CSV"
    )
    args = parser.parse_args()

    if args.song_dir:
        run_single(os.path.abspath(args.song_dir), window_size_s=args.window_size)
    else:
        run_batch(
            os.path.abspath(args.batch_dir),
            out_csv=args.out_csv,
            window_size_s=args.window_size,
        )


if __name__ == "__main__":
    main()
