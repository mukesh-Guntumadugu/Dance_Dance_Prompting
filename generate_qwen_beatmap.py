"""
generate_qwen_beatmap.py
========================
Reads pre-detected onset timestamps from `original_onsets_*.csv` files (produced
by librosa) and asks Qwen2-Audio to SELECT which onsets deserve a note and how
wide that note row should be (4 / 8 / 12 / 16 columns).

The model is NOT asked to detect onsets (already done) and NOT given note patterns
(left/right/up/down). It only decides:
  - Which of the provided onset timestamps to activate
  - What row width (4, 8, 12, or 16) to assign to each activated onset

Output per file: CSV with columns  time_ms , note_row_width

Run per song × difficulty × 6 repetitions = 30 files / song, 600 total.

Usage:
    python3 generate_qwen_beatmap.py                    # all songs, all difficulties, 6 runs
    python3 generate_qwen_beatmap.py --song "Bad Ketchup" --difficulty easy --runs 1
"""

import os
import re
import csv
import sys
import glob
import time
import datetime
import argparse
from typing import List, Optional, Tuple

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

DIFFICULTIES = ["beginner", "easy", "medium", "hard", "challenging"]

# For each difficulty: approx % of onsets to place notes on
DENSITY = {
    "beginner":    0.25,
    "easy":        0.40,
    "medium":      0.55,
    "hard":        0.70,
    "challenging": 0.90,
}

# Human-readable guidance per difficulty (used in the prompt)
DIFFICULTY_DESC = {
    "beginner":    "very sparse — only place notes on the strongest, most obvious beats. "
                   "Leave long gaps so a new player has time to react.",
    "easy":        "sparse — focus on the main beat and obvious melodic hits. "
                   "Occasional off-beat notes are fine.",
    "medium":      "moderate density — cover the main groove and most melodic events. "
                   "Include some rapid patterns but leave natural breathing room.",
    "hard":        "dense — cover nearly all rhythmically significant onsets. "
                   "Fast passages should have rapid consecutive notes.",
    "challenging": "very dense — use almost every onset. "
                   "Very fast sequences and complex rhythm patterns are expected.",
}

VALID_WIDTHS = {4, 8, 12, 16}

# ── Onset loader ──────────────────────────────────────────────────────────────

def load_original_onsets(song_dir: str) -> Optional[List[float]]:
    """Return onset timestamps (ms) from the first original_onsets_*.csv found."""
    pattern = os.path.join(song_dir, "original_onsets_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    onsets = []
    with open(matches[-1], newline="", encoding="utf-8") as f:  # use latest
        reader = csv.DictReader(f)
        for row in reader:
            try:
                onsets.append(float(row["onset_ms"]))
            except (KeyError, ValueError):
                pass
    return sorted(onsets) if onsets else None


def find_audio_file(song_dir: str) -> Optional[str]:
    """Return the first audio file found in a song directory."""
    for f in os.listdir(song_dir):
        if f.lower().endswith((".ogg", ".mp3", ".wav")):
            return os.path.join(song_dir, f)
    return None

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_beatmap_prompt(onsets_ms: List[float], difficulty: str) -> str:
    target_pct = int(DENSITY[difficulty] * 100)
    desc = DIFFICULTY_DESC[difficulty]
    onset_str = ", ".join(f"{ms:.2f}" for ms in onsets_ms)
    total = len(onsets_ms)

    return (
        f"You are a professional rhythm game chart designer creating a **{difficulty.upper()}** "
        f"difficulty beatmap for a 4-panel arrow game (like DDR/Stepmania).\n\n"

        f"## Available Onset Timestamps\n"
        f"The following {total} timestamps (in milliseconds) are musically significant moments "
        f"detected in the song. You must choose which ones deserve a note placement:\n\n"
        f"[{onset_str}]\n\n"

        f"## Your Task\n"
        f"Create a **{difficulty.upper()}** chart with approximately **{target_pct}%** of these "
        f"onsets activated ({desc}).\n\n"
        f"For each onset you choose to place a note on, output exactly ONE line:\n"
        f"  time_ms , note_row_width\n\n"

        f"## Note Row Width Rules\n"
        f"The `note_row_width` determines the rhythmic resolution of the note grid at that moment:\n"
        f"- **4**  = quarter note (1 beat in 4/4 time) — use for slow, heavy hits\n"
        f"- **8**  = eighth note (half a beat) — use for moderate rhythm patterns\n"
        f"- **12** = eighth-note triplet — use for triplet or swing rhythms\n"
        f"- **16** = sixteenth note (quarter of a beat) — use for very fast passages\n\n"
        f"Assign the width that best reflects the rhythmic feel of that moment, "
        f"consistent with the surrounding musical context.\n\n"

        f"## Output Format Rules\n"
        f"- Output ONLY the selected note lines, one per line\n"
        f"- Each line: `<time_ms> , <note_row_width>` (number, space, comma, space, number)\n"
        f"- Lines must be sorted by time_ms ascending\n"
        f"- Do NOT include headers, explanations, markdown, or any other text\n"
        f"- Only use widths from: 4, 8, 12, 16\n\n"

        f"## Example Output (first few lines of a hypothetical chart)\n"
        f"500.00 , 8\n"
        f"750.00 , 8\n"
        f"1000.00 , 4\n"
        f"1125.00 , 16\n"
        f"1250.00 , 16\n"
        f"1500.00 , 8\n\n"

        f"Now generate the {difficulty.upper()} chart:"
    )

# ── Response parser ───────────────────────────────────────────────────────────

def parse_beatmap_response(
    response_text: str,
    valid_onsets: List[float],
    tolerance_ms: float = 50.0
) -> List[Tuple[float, int]]:
    """
    Parse lines of the form `time_ms , note_row_width` from model output.
    Only keeps entries whose time_ms is within `tolerance_ms` of a valid onset.
    Returns sorted list of (time_ms, note_row_width).
    """
    valid_set = set(valid_onsets)
    results: List[Tuple[float, int]] = []

    # Match: number, comma, integer
    # e.g., "500.00 , 8"
    pattern = re.compile(r'([\d.]+)\s*,\s*(\d+)')
    matches = pattern.findall(response_text)
    
    for t_str, w_str in matches:
        try:
            t_ms = float(t_str)
            width = int(w_str)
        except ValueError:
            continue

        if width not in VALID_WIDTHS:
            continue

        # Snap to nearest valid onset within tolerance
        nearest = min(valid_onsets, key=lambda v: abs(v - t_ms))
        if abs(nearest - t_ms) <= tolerance_ms:
            results.append((nearest, width))

    # Deduplicate by time (keep first occurrence)
    seen = set()
    deduped = []
    for t, w in sorted(results):
        if t not in seen:
            seen.add(t)
            deduped.append((t, w))

    return deduped

# ── CSV saver ─────────────────────────────────────────────────────────────────

def save_beatmap_csv(
    entries: List[Tuple[float, int]],
    song_name: str,
    difficulty: str,
    run_num: int,
    out_dir: str
) -> str:
    ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    safe = song_name.replace(" ", "_").replace("/", "-")
    fname = f"Qwen_beatmap_{safe}_{difficulty}_run{run_num:02d}_{ts}.csv"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_ms", "note_row_width"])
        for t, w in entries:
            writer.writerow([t, w])
    return fpath

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate Qwen beatmaps from librosa onsets.")
    parser.add_argument("--song", type=str, default=None,
                        help="Only process this song name (substring match)")
    parser.add_argument("--difficulty", type=str, default=None,
                        choices=DIFFICULTIES,
                        help="Only process this difficulty level")
    parser.add_argument("--runs", type=int, default=6,
                        help="Number of generation runs per song × difficulty (default: 6)")
    args = parser.parse_args()

    if not os.path.isdir(BASE_DIR):
        print(f"❌ Dataset directory not found:\n   {BASE_DIR}")
        return

    # Load Qwen model once
    try:
        setup_qwen()
    except Exception as e:
        print(f"❌ Failed to load Qwen model: {e}")
        return

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d))
        and not d.startswith("_") and not d.startswith(".")
    ])

    if args.song:
        song_dirs = [s for s in song_dirs if args.song.lower() in s.lower()]

    difficulties = [args.difficulty] if args.difficulty else DIFFICULTIES

    total_files = 0
    total_expected = len(song_dirs) * len(difficulties) * args.runs

    print(f"\nGenerating Qwen beatmaps: {len(song_dirs)} songs × "
          f"{len(difficulties)} difficulty levels × {args.runs} runs "
          f"= {total_expected} files\n")
    print(f"{'Song':<40} {'Diff':<12} {'Run':<5} {'Notes':<8}  {'Output file'}")
    print("─" * 110)

    for song_name in song_dirs:
        song_dir = os.path.join(BASE_DIR, song_name)
        audio_path = find_audio_file(song_dir)
        onsets = load_original_onsets(song_dir)

        if onsets is None:
            print(f"  ⚠️  No original_onsets_*.csv found in: {song_name}")
            continue
        if audio_path is None:
            print(f"  ⚠️  No audio file found in: {song_name}")
            continue

        for difficulty in difficulties:
            prompt = build_beatmap_prompt(onsets, difficulty)

            for run in range(1, args.runs + 1):
                label = f"[{song_name[:35]:<35}] {difficulty:<10} run {run}/{args.runs}"
                print(f"  {label} ...", end="", flush=True)

                try:
                    response = generate_beatmap_with_qwen(audio_path, prompt=prompt)

                    if not response or not response.strip():
                        print(f"\n  ⚠️  Empty response — skipping")
                        continue

                    entries = parse_beatmap_response(response, onsets)

                    if not entries:
                        print(f"\n  ⚠️  No parseable entries — saving raw response")
                        ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
                        raw_path = os.path.join(
                            song_dir,
                            f"Qwen_beatmap_RAW_{song_name.replace(' ', '_')}"
                            f"_{difficulty}_run{run:02d}_{ts}.txt"
                        )
                        with open(raw_path, "w", encoding="utf-8") as f:
                            f.write(response)
                        print(f"     → {raw_path}")
                        continue

                    out_path = save_beatmap_csv(entries, song_name, difficulty, run, song_dir)
                    rel = os.path.relpath(out_path, BASE_DIR)
                    print(f"  ✅  {len(entries):>5} notes  →  {rel}")
                    total_files += 1

                except Exception as e:
                    print(f"\n  ❌  Error: {e}")

                time.sleep(0.5)

    print("─" * 110)
    print(f"\n✅  Generated {total_files}/{total_expected} beatmap files.\n")


if __name__ == "__main__":
    main()
