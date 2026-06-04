#!/usr/bin/env python3
import os
import pandas as pd

def main():
    formats = ['wav', 'mp3', 'ogg']
    
    print("=== LIBROSA PREDICTION DIFFERENCES ===")
    
    for fmt in formats:
        csv_file = f"Librosa_stateless_chunk_{fmt}_rmse.csv"
        if not os.path.exists(csv_file):
            continue
            
        df = pd.read_csv(csv_file)
        df['error'] = abs(df['pred_bpm'] - df['actual_bpm'])
        
        # Get max error per song
        song_errors = df.groupby('song_name')['error'].max()
        
        # Songs with ANY error > 0.1 BPM
        different_songs = song_errors[song_errors > 0.1].sort_index()
        
        print(f"\n[ Format: {fmt.upper()} ]")
        if len(different_songs) == 0:
            print("  -> PERFECT! Librosa predicted the exact BPM for all 50 songs.")
        else:
            print(f"  -> Librosa predicted differently from actual BPM in {len(different_songs)} / 50 songs:")
            for song, max_err in different_songs.items():
                print(f"     - {song} (Max Error: {max_err:.1f} BPM)")

if __name__ == "__main__":
    main()
