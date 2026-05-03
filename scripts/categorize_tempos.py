#!/usr/bin/env python3
"""
categorize_tempos.py
====================
Parses .sm and .ssc files to extract #BPMS and #STOPS.
Categorizes each song into tempo buckets.
"""

import os
import glob
import json
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "musicForBeatmap")
OUTPUT_JSON = os.path.join(os.path.dirname(__file__), "..", "tempo_categories.json")

def find_beatmap_files(directory):
    files = []
    for root, _, _ in os.walk(directory):
        sm_files = glob.glob(os.path.join(root, "*.sm"))
        ssc_files = glob.glob(os.path.join(root, "*.ssc"))
        basenames = set()
        for f in sm_files:
            files.append(f)
            basenames.add(os.path.splitext(os.path.basename(f))[0])
        for f in ssc_files:
            base = os.path.splitext(os.path.basename(f))[0]
            if base not in basenames:
                files.append(f)
    return files

def parse_bpms_and_stops(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    blocks = content.split(';')
    bpms, stops = [], []
    for block in blocks:
        block = block.strip()
        if block.startswith('#BPMS:'):
            data = block.split(':', 1)[1].strip()
            if data:
                for pair in data.split(','):
                    if '=' in pair:
                        beat, bpm = pair.split('=', 1)
                        try:
                            bpms.append((float(beat.strip()), float(bpm.strip())))
                        except ValueError: pass
        elif block.startswith('#STOPS:'):
            data = block.split(':', 1)[1].strip()
            if data:
                for pair in data.split(','):
                    if '=' in pair:
                        beat, slen = pair.split('=', 1)
                        try:
                            stops.append((float(beat.strip()), float(slen.strip())))
                        except ValueError: pass
    bpms.sort(key=lambda x: x[0])
    stops.sort(key=lambda x: x[0])
    return bpms, stops

def categorize_tempo(bpms, stops):
    if not bpms: return "Unknown", 0.0, 0.0, 0.0
    bpm_values = [b[1] for b in bpms]
    base_bpm, min_bpm, max_bpm = bpm_values[0], min(bpm_values), max(bpm_values)
    has_stops = len(stops) > 0
    num_changes = len(bpms) - 1
    
    if num_changes == 0 and not has_stops:
        return "Constant BPM", base_bpm, max_bpm, min_bpm
        
    is_double_half = False
    if num_changes > 0:
        for i in range(1, len(bpm_values)):
            prev, curr = bpm_values[i-1], bpm_values[i]
            if prev != 0:
                ratio = curr / prev
                if abs(ratio - 2.0) < 0.01 or abs(ratio - 0.5) < 0.01:
                    is_double_half = True
                    break
    if is_double_half:
        return "Double-time / Half-time feel", base_bpm, max_bpm, min_bpm

    if has_stops and num_changes <= 3:
        return ("Rubato / Expressive" if len(stops) > 5 else "Tempo Shift (with stops)"), base_bpm, max_bpm, min_bpm
            
    if num_changes > 0:
        diffs = [abs(bpm_values[i] - bpm_values[i-1]) for i in range(1, len(bpm_values))]
        avg_diff = sum(diffs) / len(diffs)
        if num_changes > 10 and avg_diff < 2.0:
            return "Gradual tempo change", base_bpm, max_bpm, min_bpm
        elif num_changes > 15:
            return "Rubato / Expressive", base_bpm, max_bpm, min_bpm
        else:
            return "Tempo change / BPM shift", base_bpm, max_bpm, min_bpm
            
    return "Constant BPM", base_bpm, max_bpm, min_bpm

def main():
    files = find_beatmap_files(DATA_DIR)
    categories, dataset = defaultdict(list), {}
    for file in files:
        basename = os.path.basename(file)
        bpms, stops = parse_bpms_and_stops(file)
        cat, base, mx, mn = categorize_tempo(bpms, stops)
        categories[cat].append(basename)
        dataset[basename] = {
            "file": file, "category": cat, "base_bpm": base,
            "min_bpm": mn, "max_bpm": mx, "num_bpm_changes": len(bpms) - 1,
            "num_stops": len(stops), "bpm_data": bpms, "stop_data": stops
        }
    print("\n=== TEMPO CATEGORY STATISTICS ===")
    for cat, songs in categories.items():
        print(f"{cat}: {len(songs)} songs")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(dataset, f, indent=2)
    print(f"\n✅ Finished writing tempo categorizations to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
