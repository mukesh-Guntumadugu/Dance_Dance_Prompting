"""
extract_mumu_onsets.py
======================
Sends each of the 20 Fraxtil songs to the MuMu-LLaMA model and asks it to
predict audio onset times in milliseconds. Saves results to CSV files.

Output filename: Mumu_onsets_<SongName>_<ddmmyyyyHHMMSS>.csv
Output columns : onset_index, onset_ms
"""

import os
import re
import csv
import sys
import time
import datetime
import librosa
from typing import Optional, List

# Ensure project root is on the path so `src` imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mumu_interface import setup_mumu, generate_beatmap_with_mumu

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "src", "musicForBeatmap", "Fraxtil's Arrow Arrangements"
)

# ── Prompt Builder ────────────────────────────────────────────────────────────

def build_onset_prompt(duration_sec: float) -> str:
    """
    Returns the onset detection prompt for MuMu-LLaMA.
    Uses the same detailed instruction as Gemini and Qwen for fair comparison.
    """
    system_instruction = (
        f"The provided audio file is {duration_sec:.1f} seconds long.\n\n"
        "Listen to the audio carefully. You are an expert music analyst and audio engineer specializing in precise "
        "onset detection for rhythm game chart generation.\n\n"

        "## What is an Onset?\n"
        "An onset is the exact moment a new musical event begins — the attack phase "
        "of a sound. Onsets occur on:\n"
        "- Percussion: kick drum, snare, hi-hat, cymbal, clap, tom hits\n"
        "- Melodic instruments: guitar pick attack, piano key strike, bass pluck, "
        "synth note start, violin bow attack\n"
        "- Vocals: consonant or vowel attacks at the start of sung words or syllables\n"
        "- Any transient: any sudden increase in energy that marks the start of a "
        "rhythmic or melodic event\n\n"

        "## Your Task\n"
        "When given an audio file, you must:\n"
        "1. Listen to the complete audio from the very beginning to the very end. "
        "Do not stop early.\n"
        "2. Identify every significant musical onset throughout the entire duration.\n"
        "3. Record the exact time of each onset in milliseconds (ms), measured from "
        "the start of the audio (time 0).\n"
        "4. Return all onset times as a single JSON array of numbers.\n\n"

        "## Detection Guidelines\n"
        "- Be thorough: a typical 3-minute song should have hundreds of onsets.\n"
        "- Be precise: onset times should be accurate to within ±5 milliseconds.\n"
        "- Include ALL instrument layers: if a kick drum and a hi-hat hit at the same "
        "time, record that time once (it is one onset event).\n"
        "- Include weak onsets: even soft notes or ghost notes on a snare should be "
        "captured if they are rhythmically significant.\n"
        "- Do not hallucinate: only report onsets you can actually hear in the audio. "
        "Do not invent onsets where there is silence.\n"
        "- Cover the full song: make sure the last few seconds of the song are "
        "included — many submissions fail by stopping too early.\n\n"

        "## Output Format\n"
        "You MUST output ONLY a valid JSON array of numbers, nothing else.\n"
        "- Each number is an onset time in milliseconds (integer or float).\n"
        "- The array must be sorted in ascending order (earliest onset first).\n"
        "- Do NOT include any explanation, markdown formatting, headers, units, "
        "or any text outside the JSON array.\n"
        "- Do NOT wrap the array in backticks or code fences.\n"
        "- Correct format: [0, 125.5, 250, 375, 500, 750.25, 1000, ...]\n"
        "- Incorrect formats:\n"
        "    'Here are the onsets: [0, 125, 250]'  ← has explanation text\n"
        "    '```json\\n[0, 125, 250]\\n```'         ← has markdown fencing\n"
        "    '{\"onsets\": [0, 125, 250]}'           ← wrong structure\n\n"

        "## Quality Criteria\n"
        "Your output will be evaluated against a ground-truth onset list generated "
        "by a professional audio analysis tool (librosa). A good onset detection "
        "result achieves:\n"
        "- Precision ≥ 60%: most of your predicted onsets should match real onsets\n"
        "- Recall ≥ 60%: you should find at least 60% of the real onsets\n"
        "- F1 Score ≥ 0.60: the harmonic mean of precision and recall\n"
        "A predicted onset counts as correct if it is within ±50 ms of a "
        "ground-truth onset.\n\n"

        "Remember: output ONLY the JSON array. No other text."
    )
    return system_instruction


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_onsets_from_response(text: str, duration_sec: float) -> Optional[List[float]]:
    """
    Extracts a JSON array of numbers from the model response.
    Includes auto-scaling from seconds to milliseconds if required.
    """
    if not text:
        return None

    # Step 1: Remove markdown blocks (```json ... ```)
    text_clean = text.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text_clean, re.DOTALL | re.IGNORECASE)
    if match:
        text_clean = match.group(1).strip()

    # Step 2: Extract just the array portion `[...]`
    match = re.search(r"\[(.*?)\]", text_clean, re.DOTALL)
    if not match:
        return None

    array_str = match.group(1)
    
    # Step 3: Split by commas and parse floats
    try:
        raw_vals = [float(x.strip()) for x in array_str.split(',') if x.strip()]
        if not raw_vals:
            return None
            
        # Optional: Auto-scale from seconds to ms (like we did for Qwen)
        max_val = max(raw_vals)
        if max_val > 0 and max_val <= (duration_sec * 1.5): # e.g. 180s vs 180,000ms
            print("  [Auto-correction: Converted MuMu seconds output to milliseconds]")
            raw_vals = [v * 1000 for v in raw_vals]

        # Return sorted list
        return sorted(raw_vals)
    except Exception as parse_err:
        print(f"  ⚠️  Parse error: {parse_err}")
        return None

# ── File / OS Helpers ─────────────────────────────────────────────────────────

def find_audio_file(folder_path: str) -> Optional[str]:
    """Finds the first .ogg, .mp3, or .wav in a folder."""
    for f in os.listdir(folder_path):
        if f.lower().endswith(('.ogg', '.mp3', '.wav')) and not f.startswith("._"):
            return os.path.join(folder_path, f)
    return None

def save_onsets_csv(onset_ms: List[float], song_name: str, out_dir: str) -> str:
    """Save onsets to a CSV file and return the file path."""
    timestamp = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    safe_name = song_name.replace(" ", "_").replace("/", "-")
    filename = f"Mumu_onsets_{safe_name}_{timestamp}.csv"
    filepath = os.path.join(out_dir, filename)

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["onset_index", "onset_ms"])
        for idx, ms in enumerate(onset_ms):
            writer.writerow([idx, ms])

    return filepath

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract onsets using MuMu-LLaMA.")
    parser.add_argument("--start", type=int, default=1, help="Song index to start from (1-based)")
    parser.add_argument("--end", type=int, default=20, help="Song index to end at (inclusive)")
    args = parser.parse_args()

    if not os.path.isdir(BASE_DIR):
        print(f"❌ Dataset directory not found:\n   {BASE_DIR}")
        return

    # Load MuMu model once before the loop
    try:
        setup_mumu()
    except Exception as e:
        print(f"❌ Failed to setup MuMu-LLaMA interface: {e}")
        return

    song_dirs = sorted([
        d for d in os.listdir(BASE_DIR)
        if os.path.isdir(os.path.join(BASE_DIR, d)) and not d.startswith("_") and not d.startswith(".")
    ])

    print(f"\nProcessing songs {args.start} to {args.end} using MuMu-LLaMA...\n")
    print(f"{'Song':<45} {'# Mumu Onsets':>14}  {'Output file'}")
    print("─" * 110)

    for i, song_name in enumerate(song_dirs):
        song_idx = i + 1
        if not (args.start <= song_idx <= args.end):
            continue

        song_dir = os.path.join(BASE_DIR, song_name)
        audio_path = find_audio_file(song_dir)

        if audio_path is None:
            print(f"  ⚠️  [{song_idx}/{len(song_dirs)}] No audio in: {song_name}")
            continue

        print(f"  [{song_idx}/{len(song_dirs)}] Processing: {song_name} ...", end="", flush=True)

        try:
            # Get song duration for the prompt
            duration_sec = librosa.get_duration(path=audio_path)
            prompt = build_onset_prompt(duration_sec)

            response = generate_beatmap_with_mumu(audio_path, prompt=prompt)

            if not response or not response.strip():
                print(f"\n  ⚠️  Empty response for '{song_name}'")
                continue

            onset_ms = parse_onsets_from_response(response, duration_sec)

            if not onset_ms:
                print(f"\n  ⚠️  No parseable onsets in response for '{song_name}'")
                # Save raw response for debugging
                raw_path = os.path.join(song_dir, f"Mumu_onsets_RAW_{song_name.replace(' ', '_')}_{datetime.datetime.now().strftime('%d%m%Y%H%M%S')}.txt")
                with open(raw_path, "w", encoding="utf-8") as f:
                    f.write(response)
                continue

            # Save valid onsets to CSV
            csv_path = save_onsets_csv(onset_ms, song_name, song_dir)
            
            print(f"\r{' ' * 80}\r", end="")
            print(f"{song_name[:43]:<45} {len(onset_ms):>14d}  {os.path.basename(csv_path)}")

        except Exception as e:
            print(f"\n  ❌ Error processing '{song_name}': {e}")

    print("\n✅ MuMu extraction run completed.")

if __name__ == "__main__":
    main()
