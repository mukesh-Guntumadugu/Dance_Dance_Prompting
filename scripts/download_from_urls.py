#!/usr/bin/env python3
"""
download_from_urls.py

Run on the CLUSTER. Downloads MP3s from a pixabay_urls.json file
(generated locally by collect_pixabay_urls.py).
No browser, no API key — plain requests to Pixabay CDN.

Usage:
    python scripts/download_from_urls.py \
        --urls_file pixabay_urls.json \
        --output_dir ./pixabay_music
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://pixabay.com/",
}


def sanitize(name: str, max_len: int = 80) -> str:
    return re.sub(r'[\\/:*?"<>|]', "_", name)[:max_len].strip()


def download_mp3(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"  [SKIP] {dest.name}")
        return True
    try:
        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = dest.stat().st_size / 1_000_000
        print(f"  ✓ {dest.name} ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"  [ERROR] {url}: {exc}")
        dest.unlink(missing_ok=True)
        return False


def main():
    parser = argparse.ArgumentParser(description="Download Pixabay MP3s from a collected URLs file.")
    parser.add_argument("--urls_file", type=str, default="pixabay_urls.json")
    parser.add_argument("--output_dir", type=str, default="./pixabay_music")
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between downloads (s)")
    args = parser.parse_args()

    urls_file = Path(args.urls_file)
    if not urls_file.exists():
        print(f"❌ URLs file not found: {urls_file}")
        print("   Run collect_pixabay_urls.py locally first, then copy the JSON to the cluster.")
        return

    with open(urls_file) as f:
        tracks = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    existing: dict = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            existing = json.load(f)

    print(f"Tracks in URL file : {len(tracks)}")
    print(f"Already downloaded : {len(existing)}")
    print(f"Output dir         : {output_dir}\n")

    session = requests.Session()
    session.headers.update(HEADERS)
    success = len(existing)

    for i, track in enumerate(tracks):
        tid = str(track["id"])
        if tid in existing:
            continue

        safe = sanitize(f"{track['id']}_{track['title']}")
        dest = output_dir / f"{safe}.mp3"

        print(f"↓ [{success + 1}/{len(tracks)}] {track['title']} ({track.get('duration', '?')}s)")
        ok = download_mp3(track["audio_url"], dest, session)

        if ok:
            existing[tid] = {
                "title": track["title"],
                "user": track.get("user", "Unknown"),
                "duration": track.get("duration", 0),
                "file": str(dest),
                "audio_url": track["audio_url"],
            }
            success += 1

        with open(metadata_path, "w") as f:
            json.dump(existing, f, indent=2)

        time.sleep(args.delay)

    print(f"\n✅ Done! {success}/{len(tracks)} songs in {output_dir}/")


if __name__ == "__main__":
    main()
