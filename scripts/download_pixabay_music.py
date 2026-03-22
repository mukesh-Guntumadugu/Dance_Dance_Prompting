#!/usr/bin/env python3
"""
download_pixabay_music.py

Downloads free music from Pixabay using a headless browser (Playwright)
to bypass Cloudflare protection and extract real MP3 URLs.

All Pixabay music is royalty-free.

Requirements:
    pip install playwright
    python -m playwright install chromium

Usage:
    python scripts/download_pixabay_music.py --output_dir ./pixabay_music --count 200
    python scripts/download_pixabay_music.py --output_dir ./pixabay_music --count 200 --genre electronic
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = "https://pixabay.com/music/search/"
CDN_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
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
    """Download a single MP3. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"    [SKIP] {dest.name}")
        return True
    try:
        resp = session.get(url, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                f.write(chunk)
        return True
    except Exception as exc:
        print(f"    [ERROR] {exc}")
        dest.unlink(missing_ok=True)
        return False


# ---------------------------------------------------------------------------
# Browser scraping with Playwright
# ---------------------------------------------------------------------------

def scrape_tracks_playwright(
    genre: str | None,
    order: str,
    count: int,
    headless: bool = True,
) -> list[dict]:
    """
    Use a headless Chromium browser to navigate Pixabay's music search
    and intercept the bootstrap JSON responses that contain track data.
    Returns a list of track dicts with: id, title, user, duration, audio_url.
    """
    from playwright.sync_api import sync_playwright

    tracks: list[dict] = []
    seen_ids: set = set()

    def handle_response(response):
        """Intercept bootstrap JSON responses."""
        url = response.url
        if "bootstrap" in url and url.endswith(".json"):
            try:
                data = response.json()
                # Results are under data["page"]["results"] or data["results"]
                results = (
                    data.get("page", {}).get("results", [])
                    or data.get("results", [])
                    or []
                )
                for item in results:
                    track_id = item.get("id")
                    if not track_id or track_id in seen_ids:
                        continue
                    # audio URL is under sources.src or sources[0].src
                    sources = item.get("sources", {})
                    if isinstance(sources, dict):
                        audio_url = sources.get("src", "") or sources.get("downloadUrl", "")
                    elif isinstance(sources, list) and sources:
                        audio_url = sources[0].get("src", "")
                    else:
                        audio_url = ""

                    if not audio_url:
                        # fallback: construct CDN URL from download endpoint
                        audio_url = f"https://pixabay.com/music/download/id-{track_id}.mp3"

                    seen_ids.add(track_id)
                    # Try common title keys in order
                    title = (
                        item.get("title")
                        or item.get("name")
                        or item.get("tags", "").split(",")[0].strip()
                        or f"track_{track_id}"
                    )
                    user_raw = item.get("user", "Unknown")
                    user = (
                        user_raw.get("username", "Unknown")
                        if isinstance(user_raw, dict)
                        else str(user_raw)
                    )
                    tracks.append({
                        "id": track_id,
                        "title": title,
                        "user": user,
                        "duration": item.get("duration", 0),
                        "audio_url": audio_url,
                        "page_url": f"https://pixabay.com/music/id-{track_id}/",
                    })
            except Exception:
                pass  # non-JSON or unexpected structure

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=CDN_HEADERS["User-Agent"],
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()
        page.on("response", handle_response)

        # Build URL
        params = f"?order={order}"
        if genre:
            params += f"&genre={genre}"

        page_num = 1
        while len(tracks) < count:
            url = f"{BASE_URL}{params}&pagi={page_num}"
            print(f"  [Browser] Loading page {page_num}: {url}")
            try:
                page.goto(url, timeout=30000, wait_until="networkidle")
            except Exception as exc:
                print(f"  [WARN] Page load timeout on page {page_num}: {exc}")

            # Give a moment for any lazy-loaded requests
            time.sleep(1.5)

            prev_count = len(tracks)
            if len(tracks) == prev_count and page_num > 1:
                print("  No new tracks found on this page — stopping pagination.")
                break

            print(f"  Collected {len(tracks)} tracks so far...")
            page_num += 1

            if len(tracks) >= count:
                break

        browser.close()

    return tracks[:count]


# ---------------------------------------------------------------------------
# Main download
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download free music from Pixabay using a headless browser."
    )
    parser.add_argument("--output_dir", type=str, default="./pixabay_music")
    parser.add_argument("--count", type=int, default=200, help="Number of songs to download")
    parser.add_argument(
        "--genre",
        type=str,
        default=None,
        choices=list(GENRE_MAP.keys()),
        help="Filter by genre",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="ec",
        choices=["ec", "latest", "views", "downloads"],
        help="Sort: ec=editor's choice, latest, views, downloads",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser window (non-headless) for debugging",
    )
    parser.add_argument("--delay", type=float, default=0.3, help="Delay between downloads (s)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.json"

    # Load existing metadata (for resume)
    existing: dict = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            existing = json.load(f)
    already = len(existing)
    need = args.count - already
    print(f"Target: {args.count} songs | Already downloaded: {already} | Need: {need}")

    if need <= 0:
        print("Already have enough songs!")
        return

    genre_val = GENRE_MAP.get(args.genre) if args.genre else None

    # --- Step 1: Collect track list via browser ---
    print(f"\n[1/2] Browsing Pixabay to collect {need} track URLs...")
    tracks = scrape_tracks_playwright(
        genre=genre_val,
        order=args.order,
        count=args.count,
        headless=not args.visible,
    )

    if not tracks:
        print("❌ No tracks found. Try --visible to debug the browser.")
        return

    print(f"\nFound {len(tracks)} tracks. Starting downloads...\n")

    # --- Step 2: Download MP3s ---
    session = requests.Session()
    session.headers.update(CDN_HEADERS)

    success = already
    for i, track in enumerate(tracks):
        tid = str(track["id"])
        if tid in existing:
            continue

        safe = sanitize(f"{track['id']}_{track['title']}")
        dest = output_dir / f"{safe}.mp3"

        print(f"  ↓ [{success+1}/{args.count}] {track['title']} ({track.get('duration','?')}s)")
        ok = download_mp3(track["audio_url"], dest, session)

        if ok:
            existing[tid] = {
                "title": track["title"],
                "user": track.get("user", "Unknown"),
                "duration": track.get("duration", 0),
                "file": str(dest),
                "audio_url": track["audio_url"],
                "page_url": track.get("page_url", ""),
            }
            success += 1

        # Save checkpoint
        with open(metadata_path, "w") as f:
            json.dump(existing, f, indent=2)

        time.sleep(args.delay)

    print(f"\n✅ Done! {success} songs in: {output_dir}/")
    print(f"   Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
