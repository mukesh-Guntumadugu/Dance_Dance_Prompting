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
    song_df = df[df['song_name'] == args.song].copy()

    if song_df.empty:
        print(f"No data found for {args.song} in the CSV!")
        return

    # Sort by time
    song_df = song_df.sort_values(by='window_start')

    plt.figure(figsize=(12, 6))
    
    # Plot true BPM
    plt.step(song_df['window_start'], song_df['actual_bpm'], where='post', 
             label='Actual BPM (Ground Truth)', color='blue', linewidth=2)
    
    # Plot predicted BPM
    plt.step(song_df['window_start'], song_df['pred_bpm'], where='post', 
             label='Predicted BPM', color='red', linestyle='--', linewidth=2, marker='o')

    plt.title(f"BPM Timeline Comparison for {args.song}\nModel: {os.path.basename(args.csv).split('_')[0]}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("BPM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_file = f"{args.song}_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"Graph successfully saved as {out_file}!")

if __name__ == "__main__":
    main()
