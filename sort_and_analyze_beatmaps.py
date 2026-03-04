#!/usr/bin/env python3
"""
sort_and_analyze_beatmaps.py

Three modes:

  Stage 1 — Sort:
    For every song folder in Fraxtil's Arrow Arrangements, find all .csv
    files matching a given task ID, sort rows by time_ms, and write a
    new *_sorted.csv alongside the original.

  Stage 2 — Time-Window Pattern Detection:
    Slide a 1s and 2s window across each sorted CSV and classify the
    note sequence inside each window into known DDR patterns:
        Jack, Double Step, Stream, Jump, Hand, Quad,
        Candle, Crossover, Bracket, Spin, Footswitch, Gallop, Empty
    Finds which patterns appear, how often, and which are common
    across all songs. Outputs JSON + TXT reports.

Usage:
    python sort_and_analyze_beatmaps.py                  # task0001, 1s+2s windows
    python sort_and_analyze_beatmaps.py --task task0004
    python sort_and_analyze_beatmaps.py --no-sort        # skip Stage 1
"""

import os
import csv
import json
import argparse
import datetime
from collections import Counter, defaultdict
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent / "src" / "musicForBeatmap" / "Fraxtil's Arrow Arrangements"
REPORT_DIR = Path(__file__).parent / "src" / "analysis_reports"

# Arrow indices:  L D U R  (index 0 1 2 3)
L, D, U, R = 0, 1, 2, 3
ARROW_NAMES = {0: "L", 1: "D", 2: "U", 3: "R"}

# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — SORT
# ════════════════════════════════════════════════════════════════════════════

def find_task_csvs(base_dir: Path, task_id: str) -> list[Path]:
    return sorted(
        p for p in base_dir.rglob("*.csv")
        if task_id in p.name and "_sorted" not in p.name
    )


def sort_csv_by_time(csv_path: Path) -> Path | None:
    sorted_path = csv_path.with_name(csv_path.stem + "_sorted.csv")
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    try:
        rows.sort(key=lambda r: float(r["time_ms"]))
    except (KeyError, ValueError) as e:
        print(f"  ⚠️  Cannot sort {csv_path.name}: {e}")
        return None
    with open(sorted_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return sorted_path


def stage_sort(base_dir: Path, task_id: str) -> list[Path]:
    csv_files = find_task_csvs(base_dir, task_id)
    if not csv_files:
        print(f"❌  No CSVs for task '{task_id}' under {base_dir}")
        return []
    print(f"\n{'='*62}")
    print(f"  STAGE 1 — SORT  ({len(csv_files)} files, task={task_id})")
    print(f"{'='*62}\n")
    out = []
    for p in csv_files:
        print(f"  [{p.parent.name}]  {p.name}")
        sp = sort_csv_by_time(p)
        if sp:
            print(f"    → {sp.name}")
            out.append(sp)
    print(f"\n✅  Stage 1 done — {len(out)} sorted files written.\n")
    return out


# ════════════════════════════════════════════════════════════════════════════
#  SSC / SM ORIGINAL BEATMAP PARSER
# ════════════════════════════════════════════════════════════════════════════

def _parse_bpms(bpm_str: str) -> list[tuple[float, float]]:
    """Parse BPMS field → sorted list of (beat, bpm) pairs."""
    bpms = []
    for part in bpm_str.strip().split(","):
        part = part.strip()
        if "=" not in part:
            continue
        beat_s, bpm_s = part.split("=", 1)
        try:
            bpms.append((float(beat_s), float(bpm_s)))
        except ValueError:
            pass
    return sorted(bpms, key=lambda x: x[0])


def _beat_to_ms(beat: float, bpms: list[tuple[float, float]], offset_s: float) -> float:
    """Convert a beat position to milliseconds using BPM change points."""
    time_s = -offset_s  # StepMania: negative offset = delay before song
    prev_beat, prev_bpm = bpms[0]
    for i, (chg_beat, chg_bpm) in enumerate(bpms):
        if beat <= chg_beat:
            break
        if i > 0:
            time_s += (chg_beat - prev_beat) / prev_bpm * 60.0
        prev_beat, prev_bpm = chg_beat, chg_bpm
    time_s += (beat - prev_beat) / prev_bpm * 60.0
    return round(time_s * 1000.0, 2)


def parse_ssc_chart(notes_block: str, bpms: list[tuple[float, float]],
                    offset_s: float) -> list[dict]:
    """Convert a raw NOTES block into a list of row-dicts with _time_ms and _active."""
    rows = []
    measures = notes_block.strip().split(",")
    beat = 0.0
    for measure in measures:
        note_lines = [l.strip() for l in measure.strip().splitlines()
                      if l.strip() and not l.strip().startswith("//")]
        if not note_lines:
            beat += 4.0
            continue
        rows_per_measure = len(note_lines)
        beat_per_row = 4.0 / rows_per_measure
        for line in note_lines:
            if len(line) < 4:
                beat += beat_per_row
                continue
            active = [i for i, c in enumerate(line[:4]) if c in ("1", "2", "4")]
            t_ms = _beat_to_ms(beat, bpms, offset_s)
            rows.append({
                "notes":     line[:4],
                "time_ms":   str(t_ms),
                "_time_ms":  t_ms,
                "_active":   active,
                "_n_active": len(active),
            })
            beat += beat_per_row
    # Sort by time just in case
    rows.sort(key=lambda r: r["_time_ms"])
    return rows


def parse_ssc_file(ssc_path: Path, difficulty_filter: str | None = None
                   ) -> dict[str, list[dict]]:
    """
    Parse one .ssc file.  Returns {difficulty_label: [row, ...]}.
    If difficulty_filter is given, only return that difficulty.
    """
    text = ssc_path.read_text(encoding="utf-8", errors="replace")

    # Global BPM and offset
    import re
    bpm_m   = re.search(r"#BPMS:([^;]+);", text, re.IGNORECASE)
    off_m   = re.search(r"#OFFSET:([^;]+);", text, re.IGNORECASE)
    bpms    = _parse_bpms(bpm_m.group(1)) if bpm_m else [(0.0, 120.0)]
    offset_s = float(off_m.group(1).strip()) if off_m else 0.0

    charts = {}  # difficulty → rows
    # Split on #NOTEDATA: blocks
    blocks = re.split(r"#NOTEDATA:", text, flags=re.IGNORECASE)[1:]
    for block in blocks:
        diff_m  = re.search(r"#DIFFICULTY:\s*([^;]+);", block, re.IGNORECASE)
        notes_m = re.search(r"#NOTES:\s*([\s\S]+?)(?=\n#[A-Z]|\Z)", block, re.IGNORECASE)
        if not diff_m or not notes_m:
            continue
        difficulty = diff_m.group(1).strip()
        if difficulty_filter and difficulty.lower() != difficulty_filter.lower():
            continue
        rows = parse_ssc_chart(notes_m.group(1), bpms, offset_s)
        # Only keep rows that have at least one active note
        rows = [r for r in rows if r["_active"]]
        if rows:
            key = f"{difficulty}"
            charts[key] = rows
    return charts


def load_all_original_beatmaps(base_dir: Path,
                                difficulty_filter: str | None = None,
                                song_filter: str | None = None
                                ) -> dict[str, list[dict]]:
    """
    Walk all song folders, parse their .ssc files, and return
    {"SongName (Difficulty)": [rows...]}.
    If song_filter is given, only process folders exactly matching that name.
    """
    all_charts = {}  # label → rows
    for ssc in sorted(base_dir.rglob("*.ssc")):
        song = ssc.parent.name
        if song_filter and song.lower() != song_filter.lower():
            continue
        charts = parse_ssc_file(ssc, difficulty_filter)
        for diff, rows in charts.items():
            label = f"{song} ({diff})"
            all_charts[label] = rows
            print(f"  ✓ {label}  ({len(rows)} active rows)")
    return all_charts


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — TIME-WINDOW PATTERN DETECTION
# ════════════════════════════════════════════════════════════════════════════

# ── Load rows from a sorted CSV ───────────────────────────────────────────────

_VALID_NOTE_CHARS = set("0123M")   # 0=empty 1=tap 2=hold-start 3=hold-end M=mine

def load_rows(csv_path: Path) -> list[dict]:
    """Read a sorted CSV and return rows as list of dicts, skipping separators.

    Strict validation: the 'notes' field must be exactly 4 chars of 0/1/2/3/M.
    Any other string (e.g. model prose or JSON text mistakenly written to the CSV)
    is silently skipped to prevent false pattern detections.
    """
    rows = []
    skipped = 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                notes = row.get("notes", "").strip()
                # Skip separators and empty cells
                if not notes or notes == ",":
                    continue
                # ── Strict validation ──────────────────────────────────────
                # Must be exactly 4 chars, each a valid DDR note symbol
                if len(notes) != 4 or not all(c in _VALID_NOTE_CHARS for c in notes):
                    skipped += 1
                    continue
                try:
                    row["_time_ms"] = float(row["time_ms"])
                    row["_active"] = [i for i, c in enumerate(notes) if c == "1"]
                    row["_n_active"] = len(row["_active"])
                    rows.append(row)
                except (ValueError, KeyError):
                    pass
    except Exception as e:
        print(f"  ⚠️  Error reading {csv_path.name}: {e}")
    if skipped:
        print(f"  ⚠️  {csv_path.name}: skipped {skipped} malformed note rows "
              f"(non-DDR content in notes field)")
    return rows


# ── DDR Pattern Classifiers ───────────────────────────────────────────────────
# Each classifier takes a list of active-arrow lists (one per row in the window)
# and returns True if the pattern is detected.

def is_jump(active_list: list[list[int]]) -> bool:
    """At least one row in the window has 2+ simultaneous arrows."""
    return any(len(a) >= 2 for a in active_list)


def is_jack(active_list: list[list[int]]) -> bool:
    """Same single arrow appears in two consecutive rows (rapid repeat)."""
    singles = [a[0] for a in active_list if len(a) == 1]
    return any(singles[i] == singles[i+1] for i in range(len(singles)-1))

def is_double_step(active_list: list[list[int]]) -> bool:
    """Same panel hit twice in a row with no intervening opposite foot.
    Simplified: same arrow repeated at least twice (not necessarily consecutive)."""
    singles = [a[0] for a in active_list if len(a) == 1]
    return len(singles) != len(set(singles)) and not is_jack(active_list)

def is_stream(active_list: list[list[int]]) -> bool:
    """Dense alternating single-note sequence — all singles, no repeats,
    at least 4 distinct arrows in window, all 4 columns used."""
    singles = [a[0] for a in active_list if len(a) == 1]
    if len(singles) < 4:
        return False
    return len(set(singles)) == 4 and not is_jack(active_list[:2])

def is_gallop(active_list: list[list[int]]) -> bool:
    """Un-even rhythm: a jump sandwiched between singles, or a quick 3-note burst."""
    if len(active_list) < 3:
        return False
    for i in range(len(active_list) - 2):
        a, b, c = active_list[i], active_list[i+1], active_list[i+2]
        if len(a) == 1 and len(b) >= 2 and len(c) == 1:
            return True
        if len(a) >= 2 and len(b) == 1 and len(c) >= 2:
            return True
    return False

def is_candle(active_list: list[list[int]]) -> bool:
    """Single → jump → single pattern (or jump → single → jump)."""
    if len(active_list) < 3:
        return False
    counts = [len(a) for a in active_list]
    for i in range(len(counts) - 2):
        triplet = counts[i:i+3]
        if triplet == [1, 2, 1] or triplet == [2, 1, 2]:
            return True
    return False

def active_to_side(arrow: int) -> str:
    """L/D = left foot, U/R = right foot (simplified DDR convention)."""
    return "L" if arrow in (L, D) else "R"



def is_footswitch(active_list: list[list[int]]) -> bool:
    """Alternating L-side and R-side singles in strict alternation."""
    singles = [a[0] for a in active_list if len(a) == 1]
    if len(singles) < 4:
        return False
    sides = [active_to_side(a) for a in singles]
    return all(sides[i] != sides[i+1] for i in range(len(sides)-1))

def is_bracket(active_list: list[list[int]]) -> bool:
    """Two arrows hit together that are NOT a standard jump — one foot presses two."""
    # L+D, U+R = one-foot doubles (bracket)
    for a in active_list:
        if len(a) == 2:
            pair = set(a)
            if pair == {L, D} or pair == {U, R}:
                return True
    return False

def is_spin(active_list: list[list[int]]) -> bool:
    """Circular pattern: L→D→R→U or reverse, at least 3 of 4 in sequence."""
    singles = [a[0] for a in active_list if len(a) == 1]
    if len(singles) < 3:
        return False
    clockwise    = [L, D, R, U, L, D, R, U]
    counter_cw   = [U, R, D, L, U, R, D, L]
    seq_str = "".join(str(s) for s in singles)
    for seq in [clockwise, counter_cw]:
        pat = "".join(str(s) for s in seq)
        # Check any 3-long sub-rotation appears in singles
        for i in range(len(seq) - 2):
            if "".join(str(s) for s in seq[i:i+3]) in seq_str:
                return True
    return False


# Ordered classifiers — a window is assigned the FIRST matching label
# (more specific patterns checked first)
PATTERN_CLASSIFIERS = [
    ("Bracket",     is_bracket),
    ("Spin",        is_spin),
    ("Candle",      is_candle),
    ("Gallop",      is_gallop),
    ("Jack",        is_jack),
    ("Double Step", is_double_step),
    ("Jump",        is_jump),
    ("Footswitch",  is_footswitch),
    ("Stream",      is_stream),
]

def classify_window(active_list: list[list[int]]) -> list[str]:
    """Return ALL matching pattern labels for this window (multiple can match)."""
    if not active_list or all(len(a) == 0 for a in active_list):
        return ["Empty"]
    matched = [name for name, fn in PATTERN_CLASSIFIERS if fn(active_list)]
    return matched if matched else ["Single Notes"]


# ── Sliding window ─────────────────────────────────────────────────────────---

def sliding_window_patterns(rows: list[dict], window_rows_n: int) -> Counter:
    """
    Slide a window of `window_rows_n` consecutive note-rows across the song.
    Musical meaning:
        4  rows = 1 beat (quarter-note window)   — sparse / slow patterns
        8  rows = 2 beats (half-note window)      — short bursts
        12 rows = 3 beats (triplet measure span)  — triplet patterns
        16 rows = 4 beats = 1 full measure        — full measure patterns
    For each start position, collect the next N active rows and classify them.
    Returns Counter of {pattern_name: count}.
    """
    if not rows:
        return Counter()

    pattern_counts = Counter()
    n = len(rows)

    for i in range(0, n, window_rows_n):          # jump by full window size (non-overlapping)
        # Collect window_rows_n rows starting at position i
        window = [rows[j]["_active"] for j in range(i, min(i + window_rows_n, n))]

        # Only classify windows that have at least one active note
        if any(len(a) > 0 for a in window):
            labels = classify_window(window)
            pattern_counts.update(labels)

    return pattern_counts


# ── Main Stage 2 ──────────────────────────────────────────────────────────────

# Music-notation labels for each beat-grid window size
NOTE_LABELS = {
    1:  "Whole Note Grid  (o)",
    2:  "Half Note Grid   (|o)",
    4:  "Quarter Note Grid ♩",
    8:  "Eighth Note Grid  ♪",
    12: "Triplet Grid      ♩♩♩",
    16: "Sixteenth Note Grid ♬",
}

def note_label(w: int) -> str:
    """Return the music-notation label for a window size."""
    return NOTE_LABELS.get(w, f"{w}-row grid")

def stage_analyze(source, report_dir: Path,
                  window_sizes_rows: list[int] = (4, 8, 12, 16),
                  source_label: str = "AI-generated") -> None:
    """
    source: either a list[Path] (CSV files) or dict[str, list[dict]] (pre-parsed rows).
    window_sizes_rows: list of beat-grid row counts, e.g. [4, 8, 12, 16]
        4  rows = 1 beat (quarter-note window)
        8  rows = 2 beats
        12 rows = 3 beats (triplet)
        16 rows = 1 full measure
    """
    # Normalise to dict {name: rows} and track source filenames
    if isinstance(source, dict):
        named_rows = source
        source_files = {}   # name → filename (not meaningful for SSC, use name itself)
    else:
        named_rows   = {path.name: load_rows(path) for path in source}
        source_files = {path.name: str(path) for path in source}

    n = len(named_rows)
    print(f"\n{'='*62}")
    print(f"  STAGE 2 — BEAT-GRID PATTERN DETECTION  ({source_label})")
    print(f"  Windows: {[note_label(w) for w in window_sizes_rows]}  |  {n} charts")
    print(f"{'='*62}\n")

    report_dir.mkdir(parents=True, exist_ok=True)

    # ── Per-song pattern counts per window size ───────────────────────────────
    results = {}   # name → {window_label: {pattern: count}}

    for name, rows in named_rows.items():
        r = {}
        for w in window_sizes_rows:
            label = note_label(w)
            r[label] = dict(sliding_window_patterns(rows, w).most_common())
        results[name] = r
        src = source_files.get(name, name)
        print(f"  {name}")
        print(f"    file: {src}")
        for label, counts in r.items():
            top = sorted(counts.items(), key=lambda x: -x[1])[:5]
            print(f"    [{label}] " + "  ".join(f"{p}:{n}" for p, n in top))

    # ── Cross-song: which patterns appear in how many songs ───────────────────
    cross = {}
    for w in window_sizes_rows:
        label = note_label(w)
        pat_songs = defaultdict(set)
        for song, r in results.items():
            for pat in r.get(label, {}):
                pat_songs[pat].add(song)
        cross[label] = {
            pat: {"songs": len(songs), "song_list": sorted(songs)}
            for pat, songs in sorted(pat_songs.items(), key=lambda x: -len(x[1]))
        }

    # ── Build JSON report ─────────────────────────────────────────────────────
    ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = "original" if "original" in source_label else "beatgrid"
    json_path = report_dir / f"{prefix}_patterns_{ts}.json"
    payload = {
        "generated":          ts,
        "source":             source_label,
        "window_sizes_rows":  list(window_sizes_rows),
        "window_meaning":     {
            str(w): note_label(w)
            for w in window_sizes_rows
        },
        "charts_analyzed":    n,
        "source_files":       source_files,
        "per_chart": {
            name: {
                win: sorted(counts.items(), key=lambda x: -x[1])
                for win, counts in r.items()
            }
            for name, r in results.items()
        },
        "cross_chart_prevalence": cross,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    # ── Build TXT report ──────────────────────────────────────────────────────
    txt_path = report_dir / f"{prefix}_patterns_{ts}.txt"
    lines = []
    W = 68

    # ── Pattern legend (visual arrow examples) ────────────────────────────────
    txt_path = report_dir / f"{prefix}_patterns_{ts}.txt"
    LEGEND = [
        "=" * W,
        f"  DDR-STYLE BEAT-GRID PATTERN ANALYSIS  [{source_label.upper()}]",
        f"  Generated : {ts}",
        f"  Charts    : {n}",
        f"  Windows   : {', '.join(note_label(w) for w in window_sizes_rows)}",
        f"  ♩=Quarter Note  ♪=Eighth Note  ♬=Sixteenth Note  ♩♩♩=Triplet",
        "=" * W,
        "",
        "── PATTERN LEGEND  (columns = L  D  U  R arrow positions) " + "─" * (W - 57),
        "  0 = no arrow   1 = tap   2 = hold-start   3 = hold-end   M = mine",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ QUAD  — all 4 arrows at once (use both feet + both hands)   │",
        "  │   Row:  1  1  1  1   (L D U R all active)                  │",
        "  │              ← ↓ ↑ →                                       │",
        "  │         1  1  1  1                                          │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ HAND  — 3 arrows simultaneously (3-limb hit)                │",
        "  │   Examples:  1 1 1 0  /  0 1 1 1  /  1 0 1 1  /  1 1 0 1  │",
        "  │              ← ↓ ↑ _     _ ↓ ↑ →     ← _ ↑ →     ← ↓ _ →  │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ BRACKET — one foot hits two adjacent panels simultaneously  │",
        "  │   L+D (left foot):    1 1 0 0   (← ↓ _ _)                 │",
        "  │   U+R (right foot):   0 0 1 1   (_ _ ↑ →)                 │",
        "  │   (differs from a normal jump like L+R = 1 0 0 1)          │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ SPIN  — circular rotation across 3+ consecutive rows        │",
        "  │   Clockwise:    L→D→R→U→L  (indices 0→1→3→2→0)            │",
        "  │   Row 1:  1 0 0 0  (←)                                     │",
        "  │   Row 2:  0 1 0 0  (↓)                                     │",
        "  │   Row 3:  0 0 0 1  (→)                                     │",
        "  │   Row 4:  0 0 1 0  (↑)  ← completes the spin               │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ JACK  — same single arrow in two consecutive rows (rapid    │",
        "  │         repeat, same foot hits twice fast)                  │",
        "  │   Row 1:  1 0 0 0  (←)                                     │",
        "  │   Row 2:  1 0 0 0  (←)  ← same panel again = Jack          │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ DOUBLE STEP — same arrow used again (not consecutive)       │",
        "  │   Row 1:  1 0 0 0  (←)                                     │",
        "  │   Row 2:  0 0 0 1  (→)                                     │",
        "  │   Row 3:  1 0 0 0  (←)  ← repeated, non-consecutive        │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ STREAM — alternating singles using all 4 columns, no jacks  │",
        "  │   Row 1:  1 0 0 0  (←)                                     │",
        "  │   Row 2:  0 1 0 0  (↓)                                     │",
        "  │   Row 3:  0 0 1 0  (↑)                                     │",
        "  │   Row 4:  0 0 0 1  (→)  ← 4 distinct arrows used           │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ GALLOP — quick burst: single–jump–single or jump–single–jump│",
        "  │   Row 1:  1 0 0 0  (single ←)                              │",
        "  │   Row 2:  0 1 1 0  (jump  ↓↑)                              │",
        "  │   Row 3:  0 0 0 1  (single →)                              │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ CANDLE — single→jump→single or jump→single→jump triplet     │",
        "  │   Row 1:  1 0 0 0  (single)                                │",
        "  │   Row 2:  1 0 0 1  (jump ←→)                               │",
        "  │   Row 3:  0 1 0 0  (single)                                │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ JUMP  — 2 arrows simultaneously (both feet)                 │",
        "  │   Examples:  1 0 0 1  /  0 1 1 0  /  1 0 1 0  /0 1 0 1      |",
        "  │              ← _ _ →     _ ↓ ↑ _     ← _ ↑ _               │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ CROSSOVER — foot crosses to opposite side                   │",
        "  │   Left foot hits U or R, right foot hits L or D             │",
        "  │   Row 1:  1 0 0 0  (← left foot)                           │",
        "  │   Row 2:  0 0 0 1  (→ left foot crosses over to right side) │",
        "  │   Row 3:  1 0 0 0  (← back to left side)                   │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ FOOTSWITCH — strict L-side / R-side alternation             │",
        "  │   L-side panels: L (0), D (1)                               │",
        "  │   R-side panels: U (2), R (3)                               │",
        "  │   Row 1: 1 0 0 0 (← L-side)  Row 2: 0 0 1 0 (↑ R-side)    │",
        "  │   Row 3: 0 1 0 0 (↓ L-side)  Row 4: 0 0 0 1 (→ R-side)    │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  ┌─────────────────────────────────────────────────────────────┐",
        "  │ SINGLE NOTES — isolated tap with no complex pattern context  │",
        "  │   Any row with exactly 1 active arrow, no pattern detected  │",
        "  └─────────────────────────────────────────────────────────────┘",
        "",
        "  Note: Multiple patterns can fire in the same window at once.",
        "  Counts below reflect every window where the pattern was found.",
        "─" * W,
    ]
    lines += LEGEND

    lines += [
        "",
        "── SOURCE FILES USED " + "─" * (W - 20),
    ]
    for name, filepath in (source_files.items() if source_files else {name: name for name in results}.items()):
        lines.append(f"  {name}")
        if filepath != name:
            lines.append(f"    → {filepath}")
    lines.append("─" * W)

    for w in window_sizes_rows:
        label = note_label(w)
        lines += ["", f"{'─'*W}", f"  WINDOW: {label}", f"{'─'*W}"]

        lines += ["", "  Cross-chart pattern prevalence (all patterns):", ""]
        cdata = cross[label]
        # All known pattern names + any extras found
        all_patterns = [name for name, _ in PATTERN_CLASSIFIERS] + ["Single Notes", "Empty"]
        for pat in all_patterns:
            info = cdata.get(pat, {"songs": 0})
            cnt = info["songs"]
            bar = "█" * cnt if cnt > 0 else "∅"
            absent = "" if cnt > 0 else "  (never detected)"
            lines.append(f"    {pat:<16} {bar}  ({cnt}/{n} charts){absent}")

        lines += ["", "  Per-chart patterns (all pattern types):", ""]
        for name, r in results.items():
            src = source_files.get(name, name)
            counts = r.get(label, {})  # label is already note_label(w)
            lines.append(f"    [{name}]")
            lines.append(f"      file: {src}")
            # All patterns in fixed order, with 0 for absent ones
            all_pats = [p for p, _ in PATTERN_CLASSIFIERS] + ["Single Notes", "Empty"]
            for pat in all_pats:
                c = counts.get(pat, 0)
                marker = f"×{c}" if c > 0 else "×0 (absent)"
                lines.append(f"      {pat:<16} {marker}")

    lines += ["", "=" * W]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  📄  JSON → {json_path}")
    print(f"  📄  TXT  → {txt_path}")
    print(f"\n✅  Stage 2 done.\n")


# ════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Analyse DDR-style time-window patterns in beatmaps (AI or original)."
    )
    # ── AI-generated CSV mode ─────────────────────────────────────────────
    parser.add_argument("--task",       default="task0001",
                        help="Task ID to match in CSV filenames (default: task0001)")
    parser.add_argument("--no-sort",    action="store_true",
                        help="Skip sorting — reuse already-sorted CSV files")
    parser.add_argument("--no-analyze", action="store_true",
                        help="Sort only, skip pattern analysis")
    # ── Original SSC mode ─────────────────────────────────────────────────
    parser.add_argument("--original",   action="store_true",
                        help="Analyse original .ssc beatmap files instead of AI CSVs")
    parser.add_argument("--difficulty", default=None,
                        help="(--original only) Filter to one difficulty, e.g. Hard")
    parser.add_argument("--song",       default=None,
                        help="(--original only) Filter to one specific song folder name, e.g. 'Bad Ketchup'")
    # ── Shared ────────────────────────────────────────────────────────────
    parser.add_argument("--windows",    nargs="+", type=int, default=[4, 8, 16],
                        help="Window sizes in beat-grid rows (default: 4 8 16). "
                             "4=1 beat, 8=2 beats, 12=triplet span, 16=1 full measure")
    args = parser.parse_args()

    if args.original:
        # ── Original .ssc mode ────────────────────────────────────────────
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
        label = "original"
        if args.song:
            # strip spaces from filename
            label += f"-{args.song.replace(' ', '')}"
        label += f"-{args.difficulty}" if args.difficulty else "-all"

        stage_analyze(named_rows, REPORT_DIR, args.windows, source_label=label)  # type: ignore[arg-type]

    else:
        # 
        sorted_paths = []
        if not args.no_sort:
            sorted_paths = stage_sort(BASE_DIR, args.task)
        else:
            sorted_paths = sorted(BASE_DIR.rglob(f"*{args.task}*_sorted.csv"))
            print(f"Skipping sort — using {len(sorted_paths)} existing sorted files.")

        if not sorted_paths:
            print("No sorted files to analyze.")
            return

        if not args.no_analyze:
            stage_analyze(sorted_paths, REPORT_DIR, args.windows,  # type: ignore[arg-type]
                          source_label=f"AI-{args.task}")


if __name__ == "__main__":
    main()
