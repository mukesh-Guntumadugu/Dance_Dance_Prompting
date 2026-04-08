#!/usr/bin/env python3
import os
import sys
import json
import argparse
from collections import Counter

# Map of the physical pads
PAD_MAPPING = {
    0: "Left",
    1: "Down",
    2: "Up",
    3: "Right"
}

# Delta Map for readability (Rotational Invariant moves)
DELTA_MAPPING = {
    0: "Jack (Same)",
    1: "Clockwise Step (+1)",
    2: "Crossover Jump (+2)",
    3: "Counter-Clockwise (+3)"
}

def parse_ssc_difficulties(file_path):
    """
    Parses an SSC file and extracts a list of valid rows for each difficulty.
    Returns: dict { "DifficultyName": ["0000", "0100", ...] }
    """
    charts = {}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_chart = False
    in_notes = False
    temp_difficulty = "Unknown"
    current_notes = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("#NOTEDATA:"):
            in_chart = True
            in_notes = False
            temp_difficulty = "Unknown"
            current_notes = []
            continue
            
        if in_chart:
            if line.startswith("#DIFFICULTY:"):
                # Clean tag: #DIFFICULTY:Hard;
                temp_difficulty = line.split(":")[1].replace(";", "").strip()
                
            if line.startswith("#NOTES:"):
                in_notes = True
                continue
                
            if in_notes:
                has_semicolon = ";" in line
                note_line = line.replace(";", "").split("//")[0].strip() 
                
                # Check for standard 4-key row
                if len(note_line) >= 4 and all(c in '01234M' for c in note_line[:4]):
                    current_notes.append(note_line[:4])
                
                if has_semicolon or line.startswith(";"):
                    # Chart finished
                    charts[temp_difficulty] = current_notes
                    in_chart = False
                    in_notes = False
                    
    return charts

def extract_single_tap_streams(chart_rows):
    """
    Converts note rows into isolated contiguous streams of single-taps.
    Jumps (multiple arrows) or rests (0 arrows) break the stream!
    """
    streams = []
    current_stream = []
    
    for row in chart_rows:
        # We treat a '1' (tap) or '2' (hold head) as a physical foot transition.
        active_pads = [i for i, char in enumerate(row) if char in '12']
        
        if len(active_pads) == 1:
            current_stream.append(active_pads[0])
        else:
            # Reached a jump (len > 1) or extended rest (len == 0). Break stream!
            if len(current_stream) >= 4: # We only care about streams long enough to form patterns
                streams.append(current_stream)
            current_stream = []
            
    if len(current_stream) >= 4:
        streams.append(current_stream)
        
    return streams

def calculate_chain_codes(streams):
    """
    Converts absolute pad indexes into Differential Rotational-Invariant Chain Codes.
    """
    delta_streams = []
    
    for st in streams:
        deltas = []
        for i in range(1, len(st)):
            # First-order differential calculus modulo 4
            d = (st[i] - st[i-1]) % 4
            deltas.append(d)
        delta_streams.append(deltas)
        
    return delta_streams

def extract_ngrams(delta_streams, n=3):
    """
    Extracts contiguous n-grams of delta sequences to find recurring macro-patterns.
    """
    ngrams = []
    for ds in delta_streams:
        for i in range(len(ds) - n + 1):
            pattern = tuple(ds[i:i+n])
            ngrams.append(pattern)
    return ngrams

def run_pattern_recognition(file_path, output_dir="src/pattern_recognition/human", top_k=10):
    if not os.path.exists(file_path):
        print(f"❌ Error: Could not find exactly {file_path}")
        return
        
    print(f"🎵 Analyzing Rotational Invariant Chains for: {os.path.basename(file_path)}")
    os.makedirs(output_dir, exist_ok=True)
    
    charts = parse_ssc_difficulties(file_path)
    output_json = {"file": os.path.basename(file_path), "difficulties": {}}
    
    for difficulty, rows in charts.items():
        print(f"\n--- Outputting: [ {difficulty} ] Difficulty ---")
        
        # 1. Strip raw rows to clean mechanical streams
        streams = extract_single_tap_streams(rows)
        total_stream_notes = sum(len(s) for s in streams)
        print(f"  Physical Data: {len(streams)} distinct single-tap streams detected ({total_stream_notes} total notes).")
        
        if total_stream_notes == 0:
            print("  ⚠️ No contiguous sequences found for this difficulty.")
            continue
            
        # 2. Compute Chain Codes 
        delta_streams = calculate_chain_codes(streams)
        
        # 3. Extract length-3 Chain N-Grams (representing 4 physical foot moves)
        patterns_raw = extract_ngrams(delta_streams, n=3)
        counter = Counter(patterns_raw)
        
        print("\n  Top Recognized Geometric Patterns (Invariant Sequence):")
        
        diff_output = {"total_streams": len(streams), "stream_notes": total_stream_notes, "top_patterns": []}
        
        for sequence, count in counter.most_common(top_k):
            # Translate to readable mapping
            readable = [DELTA_MAPPING[d].split(" ")[0] for d in sequence]
            seq_str = " → ".join(readable)
            raw_tup = str(sequence)
            
            pct = (count / len(patterns_raw)) * 100
            print(f"    • {raw_tup:<15} ({count:>4} occurrences | {pct:>4.1f}%) : {seq_str}")
            
            diff_output["top_patterns"].append({
                "delta_sequence": sequence,
                "label": seq_str,
                "occurrences": count,
                "percentage": round(pct, 2)
            })
            
        output_json["difficulties"][difficulty] = diff_output

    # Dump JSON Log Logically
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_file = os.path.join(output_dir, f"{base_name}_patterns.json")
    with open(out_file, "w") as f:
        json.dump(output_json, f, indent=4)
        
    print(f"\n✅ Statistical sequence log saved mathematically to: {out_file}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze Rotational Invariant chain codes inside .ssc charts")
    parser.add_argument("--ssc", type=str, default="src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ssc",
                        help="Path to .ssc human beatmap")
    parser.add_argument("--outdir", type=str, default="src/pattern_recognition/human",
                        help="Target dump directory for algorithms")
    args = parser.parse_args()
    
    run_pattern_recognition(args.ssc, args.outdir)
