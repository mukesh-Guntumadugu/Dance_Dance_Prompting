#!/usr/bin/env python3
"""
extract_osz_beatmaps.py

Batch-extracts all .osz beatmap packs and parses each .osu file inside,
pulling out metadata and hit object timestamps into JSON + CSV files.

Key: reads .osu files DIRECTLY from the zip without extracting audio,
so it runs fast even on large packs.

Usage:
    python scripts/extract_osz_beatmaps.py --input_dir ./osz_packs --output_dir ./osu_beatmaps
    python scripts/extract_osz_beatmaps.py --input_dir "/path/to/SM342 pack" --output_dir ./osu_beatmaps --columns 4
"""

import argparse
import csv
import io
import json
import zipfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_osu_text(text: str) -> dict | None:
    """
    Parse the text content of a single .osu file.

    Returns a dict with:
        metadata    – dict of key-value pairs from [General]/[Metadata]/[Difficulty]
        hit_objects – list of dicts: time_ms, time_s, column, note_type
    Returns None if not osu!mania (Mode != 3) or on parse error.
    """
    sections: dict = {
        "General": {},
        "Metadata": {},
        "Difficulty": {},
        "TimingPoints": [],
    }
    hit_objects: list[dict] = []
    current_section = None

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if line.startswith("[") and line.endswith("]"):
            current_section = line[1:-1]
            continue

        if not line or line.startswith("//"):
            continue

        if current_section in ("General", "Metadata", "Difficulty"):
            if ":" in line:
                key, _, value = line.partition(":")
                sections[current_section][key.strip()] = value.strip()

        elif current_section == "TimingPoints":
            parts = line.split(",")
            if len(parts) >= 2:
                try:
                    sections["TimingPoints"].append(
                        {"time_ms": float(parts[0]), "beat_length": float(parts[1])}
                    )
                except ValueError:
                    pass

        elif current_section == "HitObjects":
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                time_ms = int(parts[2])
                hit_type = int(parts[3])
            except ValueError:
                continue

            total_columns = int(sections["Difficulty"].get("CircleSize", "4"))
            x = int(parts[0])
            column = min(int(x * total_columns / 512), total_columns - 1)
            note_type = "long_note" if hit_type & 128 else "hit"

            hit_objects.append(
                {
                    "time_ms": time_ms,
                    "time_s": round(time_ms / 1000.0, 4),
                    "column": column,
                    "note_type": note_type,
                }
            )

    # osu!mania only
    if sections["General"].get("Mode") != "3":
        return None

    return {
        "metadata": {
            "title": sections["Metadata"].get("Title", "Unknown"),
            "artist": sections["Metadata"].get("Artist", "Unknown"),
            "creator": sections["Metadata"].get("Creator", "Unknown"),
            "version": sections["Metadata"].get("Version", "Unknown"),
            "source": sections["Metadata"].get("Source", ""),
            "audio_filename": sections["General"].get("AudioFilename", ""),
            "total_columns": int(sections["Difficulty"].get("CircleSize", "4")),
            "overall_difficulty": float(sections["Difficulty"].get("OverallDifficulty", "0")),
            "hp_drain_rate": float(sections["Difficulty"].get("HPDrainRate", "0")),
        },
        "timing_points": sections["TimingPoints"],
        "hit_objects": hit_objects,
    }


# ---------------------------------------------------------------------------
# Pack processing
# ---------------------------------------------------------------------------

def process_pack(osz_path: Path, output_dir: Path, col_filter: int | None) -> list[dict]:
    """
    Open one .osz (zip) and parse every .osu entry in-memory.
    No audio files are extracted — runs fast.
    Returns a list of summary dicts for matched beatmaps.
    """
    print(f"\n[PACK] {osz_path.name}")
    summaries: list[dict] = []

    try:
        with zipfile.ZipFile(osz_path, "r") as zf:
            osu_names = [n for n in zf.namelist() if n.lower().endswith(".osu")]
            if not osu_names:
                print("  No .osu files found – skipping.")
                return []

            for osu_name in osu_names:
                try:
                    raw_bytes = zf.read(osu_name)
                    text = raw_bytes.decode("utf-8", errors="replace")
                except Exception as exc:
                    print(f"  [WARN] Could not read {osu_name}: {exc}")
                    continue

                data = parse_osu_text(text)
                if data is None:
                    continue  # not mania or parse error

                # Apply column filter
                if col_filter is not None and data["metadata"]["total_columns"] != col_filter:
                    continue

                # Build safe filename
                safe_name = (
                    f"{data['metadata']['artist']} - {data['metadata']['title']} "
                    f"[{data['metadata']['version']}]"
                )
                safe_name = "".join(
                    c if c not in r'\/:*?"<>|' else "_" for c in safe_name
                )[:120]

                pack_output = output_dir / osz_path.stem
                pack_output.mkdir(parents=True, exist_ok=True)

                # JSON
                json_path = pack_output / f"{safe_name}.json"
                with open(json_path, "w", encoding="utf-8") as jf:
                    json.dump(data, jf, indent=2, ensure_ascii=False)

                # CSV
                csv_path = pack_output / f"{safe_name}.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as cf:
                    writer = csv.DictWriter(
                        cf, fieldnames=["time_ms", "time_s", "column", "note_type"]
                    )
                    writer.writeheader()
                    writer.writerows(data["hit_objects"])

                n = len(data["hit_objects"])
                print(
                    f"  ✓ {safe_name} | {n} notes | "
                    f"{data['metadata']['total_columns']}K"
                )

                summaries.append(
                    {
                        "pack": osz_path.stem,
                        "title": data["metadata"]["title"],
                        "artist": data["metadata"]["artist"],
                        "difficulty": data["metadata"]["version"],
                        "columns": data["metadata"]["total_columns"],
                        "note_count": n,
                        "json_path": str(json_path),
                        "csv_path": str(csv_path),
                    }
                )

    except zipfile.BadZipFile:
        print(f"  [ERROR] {osz_path.name} is not a valid zip/osz file.")

    return summaries


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch-extract osu!mania .osz packs into JSON + CSV (no audio extracted)."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="./osz_packs",
        help="Directory containing .osz files, or a folder whose CONTENTS are .osz files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./osu_beatmaps",
        help="Where to write JSON + CSV output (default: ./osu_beatmaps).",
    )
    parser.add_argument(
        "--columns",
        type=int,
        default=None,
        help="Only keep beatmaps with this key count, e.g. 4 for 4K. Default: keep all.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    osz_files = sorted(input_dir.glob("*.osz"))
    if not osz_files:
        osz_files = sorted(input_dir.glob("*.zip"))

    if not osz_files:
        print(f"No .osz files found in {input_dir}. Exiting.")
        return

    col_info = f"(filtering to {args.columns}K)" if args.columns else "(all key counts)"
    print(f"Found {len(osz_files)} pack(s) in {input_dir} {col_info}")

    all_summaries: list[dict] = []
    for osz_path in osz_files:
        all_summaries.extend(process_pack(osz_path, output_dir, args.columns))

    summary_path = output_dir / "all_beatmaps_summary.csv"
    if all_summaries:
        with open(summary_path, "w", newline="", encoding="utf-8") as sf:
            writer = csv.DictWriter(sf, fieldnames=list(all_summaries[0].keys()))
            writer.writeheader()
            writer.writerows(all_summaries)
        print(f"\n✅ Done! Extracted {len(all_summaries)} beatmap(s).")
        print(f"   Summary CSV : {summary_path}")
        print(f"   Beatmap data: {output_dir}/")
    else:
        print("\n⚠️  No matching osu!mania beatmaps found.")


if __name__ == "__main__":
    main()
