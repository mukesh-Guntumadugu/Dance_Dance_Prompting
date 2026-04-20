"""
merge_bpm_csvs.py
=================
Merges the 5 distinct model/baseline BPM Evaluation CSVs into a single 
unified matrix file using only pure built-in Python (no Pandas required).
"""

import os
import csv

def main():
    base_dir = "onsetdetection"
    models = ["LIBROSA", "QWEN", "MUMU", "DEEPRESONANCE", "FLAMINGO"]
    
    # song_data dict structure:
    # { "Song_Name": {"LIBROSA": "...", "QWEN": "...", ...} }
    song_data = {}
    found_any = False
    
    for m in models:
        path = os.path.join(base_dir, f"BPM_Estimates_{m}.csv")
        if not os.path.exists(path):
            print(f"Warning: {path} not found. Skipping...")
            continue
            
        found_any = True
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            
            # Find the value column (whichever isn't 'Song_Name', e.g. "Qwen_BPM_Estimate")
            val_cols = [c for c in reader.fieldnames if c != "Song_Name"]
            
            for row in reader:
                s_name = row["Song_Name"].strip()
                if s_name not in song_data:
                    song_data[s_name] = {}
                    
                # Store all extracted columns from this model for this song
                for col in val_cols:
                    song_data[s_name][col] = row[col]
                    
    if not found_any:
        print("No CSVs found to merge!")
        return
        
    # Compile headers dynamically based on what we found across all files
    master_headers = ["Song_Name"]
    for s_name, data_dict in song_data.items():
        for k in data_dict.keys():
            if k not in master_headers:
                master_headers.append(k)
                
    # Write the unified CSV
    out_path = os.path.join(base_dir, "Unified_BPM_Benchmark_Results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=master_headers)
        writer.writeheader()
        
        # Sort alphabetically by song name just to be safe
        for s_name in sorted(song_data.keys()):
            row = {"Song_Name": s_name}
            row.update(song_data[s_name])
            writer.writerow(row)
            
    print(f"✅ Successfully unified benchmark logs into pure python CSV:")
    print(f"   -> {out_path}")

if __name__ == "__main__":
    main()
