"""
Onset Extractor — queries the fine-tuned Qwen2-Audio server to extract
raw onset timestamps (ms) for every song in the Fraxtil pack.

Outputs a single .txt file per song in <song_dir>/qwen_onsets/ with
one millisecond timestamp per line.

Usage (after server is started):
    python3 scripts/extract_onsets_qwen.py
    python3 scripts/extract_onsets_qwen.py --server http://localhost:8000
    python3 scripts/extract_onsets_qwen.py --songs "Bad Ketchup" "Girls"
"""

import os
import sys
import re
import glob
import base64
import datetime
import argparse
import requests
import tempfile
import time
import numpy as np
import librosa
import soundfile as sf

# Ensure project root on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

SERVER_URL  = "http://localhost:8000"
BASE_DIR    = os.path.join(
    _PROJECT_ROOT, "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT   = "List the onsets in this audio segment as a comma-separated list of timestamps in seconds."


def encode_audio_chunk(y_chunk, sr) -> tuple[str, str]:
    """Write chunk to a temp wav file and return (base64, filename)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y_chunk, sr)
        tmp_path = tmp.name
    with open(tmp_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    os.unlink(tmp_path)
    return b64, "chunk.wav"


def parse_onset_response(text: str, chunk_sec: float) -> list[float]:
    """Parse comma-separated seconds from the model's output."""
    # Strip everything except digits, dots, commas
    stripped = re.sub(r'[^0-9.,]+', '', text).strip().strip(',')
    parts = [p.strip() for p in stripped.split(',') if p.strip()]
    times = []
    for p in parts:
        try:
            t = float(p)
            if 0.0 <= t <= chunk_sec + 1.0:  # Sanity: within chunk window
                times.append(t)
        except ValueError:
            pass
    return sorted(times)


def extract_onsets_for_song(song_dir: str, server_url: str, chunk_sec: float) -> tuple[list[float], float]:
    """Slice audio into chunk_sec chunks, query model, return absolute ms timestamps."""
    audio_files = (glob.glob(os.path.join(song_dir, "*.ogg")) +
                   glob.glob(os.path.join(song_dir, "*.mp3")) +
                   glob.glob(os.path.join(song_dir, "*.wav")))
    if not audio_files:
        print(f"   No audio file found — skipping.")
        return [], 0.0

    audio_path = audio_files[0]
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    all_onsets_ms = []
    total_request_time = 0.0
    chunk_starts = np.arange(0, duration, chunk_sec)

    for chunk_idx, start_sec in enumerate(chunk_starts):
        end_sec = min(start_sec + chunk_sec, duration)
        if end_sec - start_sec < 2.0:
            continue  # Skip tiny final chunk

        start_frame = int(start_sec * sr)
        end_frame   = int(end_sec   * sr)
        y_chunk     = y[start_frame:end_frame]

        audio_b64, audio_fn = encode_audio_chunk(y_chunk, sr)

        payload = {
            "audio_b64":      audio_b64,
            "audio_filename": audio_fn,
            "system_prompt":  SYSTEM_PROMPT,
            "prompt":         USER_PROMPT,   # server field is 'prompt', not 'user_prompt'
            "max_new_tokens": 512,
            "temperature":    0.0,
        }

        try:
            req_start = time.time()
            resp = requests.post(f"{server_url}/generate", json=payload, timeout=120)
            resp.raise_for_status()
            req_time = time.time() - req_start
            total_request_time += req_time
            raw_text = resp.json().get("text", "")
        except Exception as e:
            print(f"  ⚠️  Chunk {chunk_idx} failed: {e}")
            continue

        chunk_onsets_sec = parse_onset_response(raw_text, chunk_sec)
        # Convert to absolute ms
        for t_sec in chunk_onsets_sec:
            abs_ms = (start_sec + t_sec) * 1000.0
            all_onsets_ms.append(round(abs_ms, 1))

        print(f"  Chunk {chunk_idx+1}/{len(chunk_starts)}: {len(chunk_onsets_sec)} onsets detected (query time: {req_time:.2f}s)")

    return sorted(all_onsets_ms), total_request_time


def main():
    parser = argparse.ArgumentParser(description="Extract onsets using fine-tuned Qwen model")
    parser.add_argument("--server", default=SERVER_URL, help="Qwen server URL")
    parser.add_argument("--chunk_sec", type=float, default=20.0, help="Chunk size in seconds")
    parser.add_argument("--songs", nargs="+", default=None,
                        help="Specific song names to process (default: all songs)")
    args = parser.parse_args()

    # Health check
    try:
        health = requests.get(f"{args.server}/health", timeout=10).json()
        if not health.get("model_loaded"):
            print("❌ Server model not loaded yet. Start the server first.")
            sys.exit(1)
        print(f"✅ Server healthy. Constrained decoding: {health.get('constrained_decoding')}")
    except Exception as e:
        print(f"❌ Cannot reach server at {args.server}: {e}")
        sys.exit(1)

    # Find songs
    if args.songs:
        song_dirs = [os.path.join(BASE_DIR, s) for s in args.songs if os.path.isdir(os.path.join(BASE_DIR, s))]
    else:
        song_dirs = sorted([
            d for d in glob.glob(os.path.join(BASE_DIR, "*"))
            if os.path.isdir(d)
        ])

    total = len(song_dirs)
    print(f"\nProcessing {total} songs...\n")

    for idx, song_dir in enumerate(song_dirs):
        song_name = os.path.basename(song_dir)
        print(f"[{idx+1}/{total}] {song_name}")

        onsets_ms, total_request_time = extract_onsets_for_song(song_dir, args.server, args.chunk_sec)
        if not onsets_ms:
            print(f"  ⚠️  No onsets extracted for {song_name}")
            continue

        # Save output
        out_dir = os.path.join(song_dir, "qwen_onsets")
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = song_name.replace(" ", "_").replace("'", "")
        out_path = os.path.join(out_dir, f"{safe_name}_Qwen_{int(args.chunk_sec)}s_{timestamp}.csv")

        with open(out_path, "w") as f:
            f.write("onset_ms\n")
            for ms in onsets_ms:
                f.write(f"{ms}\n")

        print(f"  ✅ Saved {len(onsets_ms)} onsets → qwen_onsets/{os.path.basename(out_path)}")
        print(f"  ⏱️  Total inference time for song: {total_request_time:.2f}s\n")

    print("All songs processed!")


if __name__ == "__main__":
    main()
