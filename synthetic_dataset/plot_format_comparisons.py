#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description="Plot format comparison for a specific model")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., Librosa)")
    args = parser.parse_args()

    formats = ["wav", "mp3", "ogg"]
    dfs = {}
    
    # Load all 3 CSV files for the model
    for fmt in formats:
        csv_file = f"{args.model}_stateless_chunk_{fmt}_rmse.csv"
        if os.path.exists(csv_file):
            dfs[fmt] = pd.read_csv(csv_file)
        else:
            print(f"Warning: {csv_file} not found!")

    if not dfs:
        print("No CSV files found for this model.")
        return

    # Create output directory
    out_dir = f"{args.model}_format_comparisons"
    os.makedirs(out_dir, exist_ok=True)

    # Assume the 'wav' format has the list of songs
    if "wav" in dfs:
        song_names = dfs["wav"]['song_name'].unique()
    else:
        song_names = list(dfs.values())[0]['song_name'].unique()

    for song in song_names:
        plt.figure(figsize=(12, 6))
        
        # Plot True BPM (from any format, they all have the same true BPM)
        first_df = list(dfs.values())[0]
        song_df = first_df[first_df['song_name'] == song].sort_values(by='window_start')
        
        # Draw True BPM horizontal lines
        for i in range(len(song_df)):
            row = song_df.iloc[i]
            start = row['window_start']
            end = song_df.iloc[i+1]['window_start'] if i + 1 < len(song_df) else start + 20.0
            plt.plot([start, end], [row['actual_bpm'], row['actual_bpm']], color='black', linewidth=4, alpha=0.5)
        plt.plot([], [], color='black', linewidth=4, alpha=0.5, label='Actual BPM (Ground Truth)')

        # Plot predictions for each format
        colors = {'wav': 'blue', 'mp3': 'green', 'ogg': 'red'}
        styles = {'wav': '-', 'mp3': '--', 'ogg': ':'}
        
        for fmt, df in dfs.items():
            fmt_song_df = df[df['song_name'] == song].sort_values(by='window_start')
            
            for i in range(len(fmt_song_df)):
                row = fmt_song_df.iloc[i]
                start = row['window_start']
                end = fmt_song_df.iloc[i+1]['window_start'] if i + 1 < len(fmt_song_df) else start + 20.0
                plt.plot([start, end], [row['pred_bpm'], row['pred_bpm']], color=colors[fmt], linestyle=styles[fmt], linewidth=2)
            
            # Proxy artist for legend
            plt.plot([], [], color=colors[fmt], linestyle=styles[fmt], linewidth=2, label=f'{args.model} ({fmt.upper()})')

        plt.title(f"Audio Format Comparison for {song}\nModel: {args.model}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("BPM")
        plt.xlim(0, 250)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_file = os.path.join(out_dir, f"{args.model}_format_comp_{song}.png")
        plt.savefig(out_file, dpi=300)
        plt.close()

    print(f"Successfully generated {len(song_names)} format comparison graphs in '{out_dir}/'!")

if __name__ == "__main__":
    main()
