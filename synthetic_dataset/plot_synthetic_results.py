#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot actual vs predicted BPM from synthetic evaluation")
    parser.add_argument("--csv", type=str, required=True, help="Path to the RMSE CSV file (e.g., Qwen_stateless_chunk_rmse.csv)")
    parser.add_argument("--song", type=str, default="song_50", help="Which song to plot (e.g., song_50)")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"Error: Could not find {args.csv}")
        print("Make sure you downloaded it from the HPC!")
        return

    df = pd.read_csv(args.csv)
    song_names = df['song_name'].unique() if args.song == "all" else [args.song]
    
    for current_song in song_names:
        song_df = df[df['song_name'] == current_song].copy()
    
        if song_df.empty:
            print(f"No data found for {current_song} in the CSV!")
            continue
    
        # Sort by time
        song_df = song_df.sort_values(by='window_start')
    
        plt.figure(figsize=(12, 6))
        
        # Plot true BPM
        plt.plot(song_df['window_start'], song_df['actual_bpm'], 
                 label='Actual BPM (Ground Truth)', color='blue', linewidth=2, marker='o')
        
        # Plot predicted BPM
        plt.plot(song_df['window_start'], song_df['pred_bpm'], 
                 label='Predicted BPM', color='red', linestyle='--', linewidth=2, marker='o')
    
        base_name = os.path.basename(args.csv)
        model_name = base_name.split('_')[0]
        parts = base_name.split('_')
        fmt = parts[-2] if len(parts) >= 4 and parts[-2] in ["wav", "mp3", "ogg"] else "wav"
        
        plt.title(f"BPM Timeline Comparison for {current_song}\nModel: {model_name} (Format: {fmt.upper()})")
        plt.xlabel("Time (seconds)")
        plt.ylabel("BPM")
        plt.xlim(0, 250) # Lock the X-axis to 250s max
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
    
        out_file = f"{model_name}_{fmt}_{current_song}_plot.png"
        plt.savefig(out_file, dpi=300)
        plt.close()
        print(f"Graph successfully saved as {out_file}!")

if __name__ == "__main__":
    main()
