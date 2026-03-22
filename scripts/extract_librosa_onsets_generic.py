#!/usr/bin/env python3
"""
extract_librosa_onsets_generic.py

Generic version of extract_librosa_onsets.py.
Works on ANY folder of audio files, not just the Fraxtil dataset.

Processes all .mp3/.ogg/.wav files in --audio_dir and saves
original_onsets_<name>.csv next to each audio file.

Usage:
    python scripts/extract_librosa_onsets_generic.py --audio_dir ./pixabay_music
"""

import argparse
import csv
import datetime
import os
import re
from pathlib import Path

import librosa


def sanitize(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)


def detect_onsets_ms(audio_path: str) -> list[float]:
    y, sr = librosa.load(audio_path, sr=None)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env,
        sr=sr,
        backtrack=True,
        units="frames",
    )
    onset_times_sec = librosa.frames_to_time(onset_frames, sr=sr)
    return [round(float(t) * 1000, 2) for t in onset_times_sec]


def save_onsets_csv(onset_ms: list[float], out_path: Path) -> None:
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_index", "onset_ms"])
        for idx, ms in enumerate(onset_ms):
            writer.writerow([idx, ms])


def main():
    parser = argparse.ArgumentParser(description="Extract Librosa onsets from a folder of audio files.")
    parser.add_argument("--audio_dir", type=str, required=True, help="Folder containing MP3/OGG/WAV files")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        print(f"❌ Directory not found: {audio_dir}")
        return

    audio_files = sorted(
        f for f in audio_dir.iterdir()
        if f.suffix.lower() in (".mp3", ".ogg", ".wav", ".flac")
    )

    if not audio_files:
        print(f"No audio files found in {audio_dir}")
        return

    print(f"Found {len(audio_files)} audio files in {audio_dir}\n")
    print(f"{'File':<60} {'Onsets':>8}")
    print("─" * 72)

    total_onsets = 0
    processed = 0

    for audio_path in audio_files:
        # Skip if already extracted
        csv_path = audio_dir / f"original_onsets_{sanitize(audio_path.stem)}.csv"
        if csv_path.exists():
            print(f"  [SKIP] {audio_path.name} (already extracted)")
            processed += 1
            continue

        try:
            onset_ms = detect_onsets_ms(str(audio_path))
            save_onsets_csv(onset_ms, csv_path)
            print(f"  {audio_path.name:<58} {len(onset_ms):>8,}")
            total_onsets += len(onset_ms)
            processed += 1
        except Exception as exc:
            print(f"  [ERROR] {audio_path.name}: {exc}")

    print("─" * 72)
    print(f"\n✅ Processed {processed}/{len(audio_files)} files | Total onsets: {total_onsets:,}")


if __name__ == "__main__":
    main()
