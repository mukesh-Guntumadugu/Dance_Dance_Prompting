#!/usr/bin/env python3
"""
extract_rhythm_density.py
=========================
Parses .sm and .ssc files to extract the frequency of 4th, 8th, 12th, 16th, etc. notes
for each difficulty level in 'dance-single' mode. Prioritizes .sm over .ssc.
Outputs a CSV with density percentages for correlation analysis.
"""

import os
import glob
import csv
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "musicForBeatmap")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "..", "rhythm_density_stats.csv")

def find_beatmap_files(directory):
    """Finds SM and SSC files, prioritizing SM over SSC as requested."""
    files = []
    for root, _, _ in os.walk(directory):
        sm_files = glob.glob(os.path.join(root, "*.sm"))
        ssc_files = glob.glob(os.path.join(root, "*.ssc"))
        
        # Priority: SM first, then SSC if SM doesn't exist for that basename
        basenames = set()
        for f in sm_files:
            files.append(f)
            basenames.add(os.path.splitext(os.path.basename(f))[0])
            
        for f in ssc_files:
            base = os.path.splitext(os.path.basename(f))[0]
            if base not in basenames:
                files.append(f)
                
    return files

def get_active_notes_in_line(line):
    """Returns True if there is an active tap/hold/roll on this line."""
    # 1=Tap, 2=Hold start, 4=Roll start
    return '1' in line or '2' in line or '4' in line

def parse_measure(measure_lines):
    """
    Parses a single measure and returns a dict of counts for each quantization.
    """
    num_lines = len(measure_lines)
    counts = defaultdict(int)
    
    for i, line in enumerate(measure_lines):
        if not get_active_notes_in_line(line):
            continue
            
        fraction = i / num_lines
        
        if fraction % 0.25 == 0:
            counts["4th"] += 1
        elif fraction % 0.125 == 0:
            counts["8th"] += 1
        elif fraction % (1/12) < 0.0001 or fraction % (1/12) > (1/12)-0.0001:
            counts["12th"] += 1
        elif fraction % 0.0625 == 0:
            counts["16th"] += 1
        elif fraction % (1/24) < 0.0001 or fraction % (1/24) > (1/24)-0.0001:
            counts["24th"] += 1
        elif fraction % 0.03125 == 0:
            counts["32nd"] += 1
        elif fraction % 0.015625 == 0:
            counts["64th"] += 1
        else:
            counts["other"] += 1
            
    return counts

def get_difficulty_label(counts):
    """Categorizes difficulty based on note density."""
    tot = counts.get("total", 0)
    if tot == 0: return "Unknown"
    
    p16 = (counts.get("16th", 0) / tot) * 100
    p24_plus = ((counts.get("24th", 0) + counts.get("32nd", 0) + counts.get("64th", 0)) / tot) * 100
    p4_8 = ((counts.get("4th", 0) + counts.get("8th", 0)) / tot) * 100
    
    if p24_plus > 5 or p16 > 30:
        return "Challenging"
    elif p16 > 10 or p24_plus > 0:
        return "Hard"
    elif p4_8 > 80:
        return "Easy"
    else:
        return "Medium"

def parse_notes_data(notes_data):
    """Splits notes data into measures and counts quantization."""
    totals = defaultdict(int)
    measures = notes_data.split(',')
    for measure in measures:
        lines = [l.strip() for l in measure.split('\n')
                 if l.strip() and not l.strip().startswith('//') and len(l.strip()) == 4]
        if not lines:
            continue
        measure_counts = parse_measure(lines)
        for k, v in measure_counts.items():
            totals[k] += v
            totals["total"] += v
    return totals

def parse_beatmap(filepath):
    """Parses an SM or SSC file."""
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read().replace('\r\n', '\n').replace('\r', '\n')
    
    results = {}
    is_ssc = filepath.endswith('.ssc')
    
    if not is_ssc:
        blocks = content.split(';')
        for block in blocks:
            if '#NOTES:' not in block: continue
            notes_idx = block.find('#NOTES:')
            block = block[notes_idx:]
            lines = block.split('\n')
            if len(lines) < 7: continue
            stepstype = lines[1].strip().rstrip(':')
            difficulty = lines[3].strip().rstrip(':')
            if stepstype == 'dance-single':
                note_data_str = '\n'.join(lines[6:])
                totals = parse_notes_data(note_data_str)
                if totals.get('total', 0) > 0:
                    results[difficulty] = totals
    else:
        current_stepstype = ""
        current_difficulty = ""
        in_notedata = False
        blocks = content.split(';')
        for block in blocks:
            block = block.strip()
            if not block: continue
            if block.startswith('#NOTEDATA:'):
                in_notedata = True
                current_stepstype = ""
                current_difficulty = ""
            elif block.startswith('#STEPSTYPE:'):
                current_stepstype = block.split(':', 1)[1].strip()
            elif block.startswith('#DIFFICULTY:'):
                current_difficulty = block.split(':', 1)[1].strip()
            elif block.startswith('#NOTES:') and in_notedata:
                if current_stepstype == 'dance-single':
                    note_data_str = block.split(':', 1)[1].strip()
                    totals = parse_notes_data(note_data_str)
                    if totals.get('total', 0) > 0:
                        results[current_difficulty] = totals
                in_notedata = False
    return results

def main():
    print(f"Scanning for beatmaps in {DATA_DIR}...")
    files = find_beatmap_files(DATA_DIR)
    print(f"Found {len(files)} unique beatmaps.")
    
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Song", "File_Type", "Difficulty", "Total_Notes", 
                         "4th_%", "8th_%", "12th_%", "16th_%", "24th_%", "32nd_%", "64th_%", "Other_%", "Density_Label"])
        
        for file in files:
            basename = os.path.basename(file)
            ext = os.path.splitext(file)[1]
            try:
                data = parse_beatmap(file)
                for diff, counts in data.items():
                    tot = counts.get("total", 0)
                    if tot == 0: continue
                        
                    p4 = (counts.get("4th", 0) / tot) * 100
                    p8 = (counts.get("8th", 0) / tot) * 100
                    p12 = (counts.get("12th", 0) / tot) * 100
                    p16 = (counts.get("16th", 0) / tot) * 100
                    p24 = (counts.get("24th", 0) / tot) * 100
                    p32 = (counts.get("32nd", 0) / tot) * 100
                    p64 = (counts.get("64th", 0) / tot) * 100
                    po = (counts.get("other", 0) / tot) * 100
                    label = get_difficulty_label(counts)
                    
                    writer.writerow([basename, ext, diff, tot, 
                                     f"{p4:.1f}", f"{p8:.1f}", f"{p12:.1f}", f"{p16:.1f}", 
                                     f"{p24:.1f}", f"{p32:.1f}", f"{p64:.1f}", f"{po:.1f}", label])
            except Exception as e:
                print(f"Error parsing {basename}: {e}")
                
    print(f"✅ Finished writing {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
