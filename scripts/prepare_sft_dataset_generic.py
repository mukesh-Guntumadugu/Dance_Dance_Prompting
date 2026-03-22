#!/usr/bin/env python3
"""
prepare_sft_dataset_generic.py

Generic version of prepare_sft_dataset.py.
Works on ANY folder of (audio, original_onsets_*.csv) pairs.

Usage:
    python scripts/prepare_sft_dataset_generic.py \
        --audio_dir ./pixabay_music \
        --output_dir ./sft_dataset_pixabay
"""

import argparse
import glob
import json
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

CHUNK_DURATION = 20.0  # seconds per training sample


def parse_onsets_csv(csv_path: str) -> list[float]:
    """Parse original_onsets_*.csv → list of onset times in seconds."""
    onsets_sec = []
    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    ms = float(parts[1])
                    onsets_sec.append(round(ms / 1000.0, 3))
                except ValueError:
                    pass
    except Exception as exc:
        print(f"  [WARN] Could not parse {csv_path}: {exc}")
    return sorted(onsets_sec)


def main():
    parser = argparse.ArgumentParser(description="Build SFT dataset from audio + onset CSV pairs.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Folder with MP3s and original_onsets_*.csv files")
    parser.add_argument("--output_dir", type=str, default="./sft_dataset_pixabay")
    parser.add_argument("--chunk_duration", type=float, default=CHUNK_DURATION)
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)
    audio_out_dir = output_dir / "audio"
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    # Find all audio files that have a matching onset CSV
    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in (".mp3", ".ogg", ".wav", ".flac")
    )

    jsonl_data = []

    for audio_path in audio_files:
        # Find matching onset CSV: original_onsets_<stem>.csv
        csv_pattern = str(audio_dir / f"original_onsets_{audio_path.stem}.csv")
        onset_files = glob.glob(csv_pattern)

        # Also try sanitized name
        if not onset_files:
            import re
            safe_stem = re.sub(r'[\\/:*?"<>|]', "_", audio_path.stem)
            csv_pattern2 = str(audio_dir / f"original_onsets_{safe_stem}.csv")
            onset_files = glob.glob(csv_pattern2)

        if not onset_files:
            print(f"  [SKIP] No onset CSV for: {audio_path.name}")
            continue

        onsets = parse_onsets_csv(onset_files[0])
        if not onsets:
            print(f"  [SKIP] Empty onsets for: {audio_path.name}")
            continue

        print(f"  Processing: {audio_path.name} ({len(onsets)} onsets)")

        try:
            y, sr = librosa.load(str(audio_path), sr=None)
        except Exception as exc:
            print(f"  [ERROR] Could not load {audio_path.name}: {exc}")
            continue

        duration = librosa.get_duration(y=y, sr=sr)
        song_id = audio_path.stem[:40].replace(" ", "_")

        for chunk_idx, start_time in enumerate(np.arange(0, duration, args.chunk_duration)):
            end_time = min(start_time + args.chunk_duration, duration)
            if end_time - start_time < 5.0:
                continue  # skip very short final chunk

            chunk_onsets = [
                round(o - start_time, 3)
                for o in onsets
                if start_time <= o < end_time
            ]
            if not chunk_onsets:
                continue

            # Save audio chunk
            start_frame = int(start_time * sr)
            end_frame = int(end_time * sr)
            y_chunk = y[start_frame:end_frame]
            chunk_filename = f"{song_id}_{chunk_idx:03d}.wav"
            chunk_path = audio_out_dir / chunk_filename
            sf.write(str(chunk_path), y_chunk, sr)

            onset_str = ", ".join(map(str, chunk_onsets))
            entry = {
                "id": f"{song_id}_{chunk_idx:03d}",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": str(chunk_path.resolve())},
                            {"type": "text", "text": "List the onsets in this audio segment as a comma-separated list of timestamps in seconds."},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": onset_str}],
                    },
                ],
            }
            jsonl_data.append(entry)

    jsonl_path = output_dir / "dataset.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in jsonl_data:
            f.write(json.dumps(entry) + "\n")

    print(f"\n✅ Created {len(jsonl_data)} training samples.")
    print(f"   Dataset : {jsonl_path}")
    print(f"   Audio   : {audio_out_dir}/")


if __name__ == "__main__":
    main()
