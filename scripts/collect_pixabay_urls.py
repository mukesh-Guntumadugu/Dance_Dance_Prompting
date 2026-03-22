#!/usr/bin/env python3
"""
collect_pixabay_urls.py

Run LOCALLY. Uses Playwright browser to scrape Pixabay music search pages
and saves CDN audio URLs to pixabay_urls.json.
No API key required.

Then copy pixabay_urls.json to the cluster and run:
    python scripts/download_from_urls.py --urls_file pixabay_urls.json --output_dir ./pixabay_music

Usage:
    python scripts/collect_pixabay_urls.py --count 100 --output pixabay_urls.json
"""

import argparse
import json
import time
from pathlib import Path


def collect_urls(count: int, order: str, genre: str | None, headless: bool) -> list[dict]:
    from playwright.sync_api import sync_playwright

    tracks: list[dict] = []
    seen_ids: set = set()

    def handle_response(response):
        url = response.url
        if "bootstrap" not in url or not url.endswith(".json"):
            return
        try:
            data = response.json()
            results = (
                data.get("page", {}).get("results", [])
                or data.get("results", [])
                or []
            )
            for item in results:
                track_id = item.get("id")
                if not track_id or track_id in seen_ids:
                    continue

                sources = item.get("sources", {})
                if isinstance(sources, dict):
                    audio_url = sources.get("src", "") or sources.get("downloadUrl", "")
                elif isinstance(sources, list) and sources:
                    audio_url = sources[0].get("src", "")
                else:
                    audio_url = ""

                if not audio_url:
                    audio_url = f"https://pixabay.com/music/download/id-{track_id}.mp3"

                title = (
                    item.get("title")
                    or item.get("name")
                    or (item.get("tags", "").split(",")[0].strip() if item.get("tags") else "")
                    or f"track_{track_id}"
                )
                user_raw = item.get("user", {})
                user = user_raw.get("username", "Unknown") if isinstance(user_raw, dict) else str(user_raw)

                seen_ids.add(track_id)
                tracks.append({
                    "id": track_id,
                    "title": title,
                    "user": user,
                    "duration": item.get("duration", 0),
                    "audio_url": audio_url,
                    "page_url": f"https://pixabay.com/music/id-{track_id}/",
                })
                print(f"  [{len(tracks)}] {title} ({item.get('duration', '?')}s)")
        except Exception:
            pass

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 900},
        )
        page = context.new_page()
        page.on("response", handle_response)

        params = f"?order={order}"
        if genre:
            params += f"&genre={genre}"

        page_num = 1
        while len(tracks) < count:
            url = f"https://pixabay.com/music/search/{params}&pagi={page_num}"
            print(f"\n[PAGE {page_num}] {url}")
            try:
                page.goto(url, timeout=30000, wait_until="networkidle")
            except Exception as exc:
                print(f"  Timeout on page {page_num}: {exc}")
            time.sleep(2)

            if page_num > 1 and len(tracks) == 0:
                print("No tracks found at all — check if Pixabay changed layout.")
                break

            page_num += 1

        browser.close()

    return tracks[:count]


def main():
    parser = argparse.ArgumentParser(description="Collect Pixabay music URLs using a headless browser.")
    parser.add_argument("--count", type=int, default=100, help="Number of tracks to collect")
    parser.add_argument("--output", type=str, default="pixabay_urls.json")
    parser.add_argument("--order", type=str, default="ec", choices=["ec", "latest", "downloads"])
    parser.add_argument("--genre", type=str, default=None)
    parser.add_argument("--visible", action="store_true", help="Show browser window for debugging")
    args = parser.parse_args()

    print(f"Collecting {args.count} Pixabay track URLs (no download yet)...")
    tracks = collect_urls(
        count=args.count,
        order=args.order,
        genre=args.genre,
        headless=not args.visible,
    )

    out = Path(args.output)
    with open(out, "w") as f:
        json.dump(tracks, f, indent=2)

    print(f"\n✅ Saved {len(tracks)} track URLs to: {out}")
    print(f"   Copy this file to the cluster, then run:")
    print(f"   python scripts/download_from_urls.py --urls_file {out} --output_dir ./pixabay_music")


if __name__ == "__main__":
    main()
