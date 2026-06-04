#!/usr/bin/env python3
import json
import os
import matplotlib.pyplot as plt

def main():
    formats = ['wav', 'mp3', 'ogg']
    colors = {'wav': 'green', 'mp3': 'blue', 'ogg': 'orange'}
    
    plt.figure(figsize=(15, 8))
    plt.title("Librosa BPM Prediction Error by Audio Format (60 - 240 BPM)", fontsize=16)
    plt.xlabel("Actual Ground Truth BPM", fontsize=14)
    plt.ylabel("RMSE (Error in BPM)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    for fmt in formats:
        json_path = f"Librosa_full_song_{fmt}_report.json"
        if not os.path.exists(json_path):
            print(f"Skipping {fmt}, file not found.")
            continue
            
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        x_bpm = []
        y_err = []
        
        # Sort keys to ensure line plots linearly from 60 to 240
        for song_name in sorted([k for k in data.keys() if k.startswith('bpm_')], key=lambda k: int(k.split('_')[1])):
            bpm_val = int(song_name.split('_')[1])
            err = data[song_name]
            if err is not None:
                x_bpm.append(bpm_val)
                y_err.append(err)
                
        if x_bpm and y_err:
            plt.plot(x_bpm, y_err, label=f"Librosa ({fmt.upper()})", color=colors[fmt], linewidth=2, marker='o', markersize=3, alpha=0.8)

    plt.legend(fontsize=12)
    plt.xticks(range(60, 241, 10))
    
    # Highlight perfect zone
    plt.axhspan(0, 5, color='green', alpha=0.1, label="Acceptable Error (< 5 BPM)")
    
    plt.tight_layout()
    out_path = "Librosa_BPM_Boundary_Sweep.png"
    plt.savefig(out_path, dpi=300)
    print(f"Successfully generated sweep graph: {out_path}")

if __name__ == "__main__":
    main()
