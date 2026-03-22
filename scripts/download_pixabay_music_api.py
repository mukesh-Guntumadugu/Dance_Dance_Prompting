#!/usr/bin/env python3
"""
download_pixabay_music_api.py

Downloads free music from Pixabay using their official API.
Works on HPC clusters — no browser or Playwright required.

Get a FREE API key (takes 30 sec) at: https://pixabay.com/api/docs/

Usage:
    # With API key (recommended)
    python scripts/download_pixabay_music_api.py --api_key YOUR_KEY --count 100

    # Set key as env var instead
    export PIXABAY_KEY=your_key_here
    python scripts/download_pixabay_music_api.py --count 100
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_URL = "https://pixabay.com/api/videos/"   # Pixabay uses /videos/ for music too
MUSIC_API_URL = "https://pixabay.com/api/music/"  # direct music endpoint (if available)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Referer": "https://pixabay.com/",
}

GENRE_MAP = {
    "electronic": "electronic",
    "rock": "rock",
    "jazz": "jazz",
    "classical": "classical",
    "ambient": "ambient",
    "pop": "pop",
    "hiphop": "hip-hop",
    "cinematic": "cinematic",
    "folk": "folk-acoustic",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sanitize(name: str, max_len: int = 80) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name[:max_len].strip()


def download_mp3(url: str, dest: Path, session: requests.Session) -> bool:
    if dest.exists() and dest.stat().st_size > 10_000:
        print(f"    [SKIP] Already exists: {dest.name}")
        return True
    try:
        resp = session.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        size_mb = dest.stat().st_size / 1_000_000
        print(f"    ✓ {dest.name} ({size_mb:.1f} MB)")
        return True
    except Exception as exc:
        print(f"    [ERROR] {exc}")
        dest.unlink(missing_ok=True)
        return False


def fetch_page(api_key: str, page: int, per_page: int, order: str, genre: str | None) -> list[dict]:
    """Fetch one page of tracks from the Pixabay music API."""
    params = {
        "key": api_key,
        "per_page": per_page,
        "page": page,
        "order": order,
    }
    if genre:
        params["category"] = genre

    session = requests.Session()
    session.headers.update(HEADERS)

    # Try the dedicated music endpoint first
    try:
        resp = session.get(MUSIC_API_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("hits", [])
        if hits:
            return hits
    except Exception:
        pass

    return []


def parse_hit(hit: dict, fallback_id: int) -> dict | None:
    """Extract usable track info from a Pixabay API hit."""
    # Audio URL can be in multiple places depending on API version
    audio_url = (
        hit.get("audio", {}).get("url", "") if isinstance(hit.get("audio"), dict)
        else hit.get("audio", "")
        or hit.get("url", "")
    )
    if not audio_url:
        # Construct a Pixabay download URL
        track_id = hit.get("id", fallback_id)
        audio_url = f"https://pixabay.com/music/download/id-{track_id}.mp3"

    title = (
        hit.get("title")
        or hit.get("name")
        or hit.get("tags", "").split(",")[0].strip()
        or f"track_{hit.get('id', fallback_id)}"
    )
    user = hit.get("user", "Unknown")

    return {
        "id": hit.get("id", fallback_id),
        "title": title,
        "user": user,
        "duration": hit.get("duration", 0),
        "audio_url": audio_url,
        "page_url": f"https://pixabay.com/music/id-{hit.get('id', fallback_id)}/",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download Pixabay music via API (cluster-friendly, no browser required)."
    )
    parser.add_argument("--output_dir", type=str, default="./pixabay_music")
    parser.add_argument("--count", type=int, default=100, help="Number of songs to download")
    parser.add_argument(
        "--api_key",
        type=str,
        default=os.environ.get("PIXABAY_KEY", ""),
        help="Pixabay API key. Get free at https://pixabay.com/api/docs/ or set PIXABAY_KEY env var.",
    )
    parser.add_argument(
        "--genre",
        type=str,
        default=None,
        choices=list(GENRE_MAP.keys()),
        help="Filter by genre (optional)",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="ec",
        choices=["ec", "latest", "popular"],
        help="Sort order: ec=editor's choice (default)",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between downloads (s)")
    args = parser.parse_args()

    if not args.api_key:
        print("❌ No API key provided.")
        print("   Get a FREE key in 30 seconds at: https://pixabay.com/api/docs/")
        print("   Then run: python scripts/download_pixabay_music_api.py --api_key YOUR_KEY")
        print("\n   Or on the cluster: sbatch slurm_download_pixabay.sh --export=ALL,PIXABAY_KEY=YOUR_KEY")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    existing: dict = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            existing = json.load(f)

    already = len(existing)
    print(f"Target: {args.count} | Already downloaded: {already} | Need: {args.count - already}")

    session = requests.Session()
    session.headers.update(HEADERS)

    success = already
    page = 1
    per_page = min(200, args.count)
    genre_val = GENRE_MAP.get(args.genre) if args.genre else None

    while success < args.count:
        print(f"\n[PAGE {page}] Fetching track list...")
        hits = fetch_page(args.api_key, page, per_page, args.order, genre_val)

        if not hits:
            print("  No more tracks available.")
            break

        for i, hit in enumerate(hits):
            if success >= args.count:
                break

            track = parse_hit(hit, i)
            if not track:
                continue

            tid = str(track["id"])
            if tid in existing:
                continue

            safe = sanitize(f"{track['id']}_{track['title']}")
            dest = output_dir / f"{safe}.mp3"

            print(f"  ↓ [{success+1}/{args.count}] {track['title']} ({track.get('duration','?')}s) — by {track['user']}")
            ok = download_mp3(track["audio_url"], dest, session)

            if ok:
                existing[tid] = {
                    "title": track["title"],
                    "user": track["user"],
                    "duration": track.get("duration", 0),
                    "file": str(dest),
                    "audio_url": track["audio_url"],
                    "page_url": track.get("page_url", ""),
                }
                success += 1

            with open(metadata_path, "w") as f:
                json.dump(existing, f, indent=2)

            time.sleep(args.delay)

        page += 1

    print(f"\n✅ Done! {success} songs saved to: {output_dir}/")
    print(f"   Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
