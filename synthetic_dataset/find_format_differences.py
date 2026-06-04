#!/usr/bin/env python3
import os
import pandas as pd
import json

def main():
    formats = ['wav', 'mp3', 'ogg']
    reports = {}
    
    for fmt in formats:
        json_file = f"Librosa_stateless_chunk_{fmt}_report.json"
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                reports[fmt] = json.load(f)
                
    if len(reports) < 3:
        print("Missing some report JSON files!")
        return

    print("=== FORMAT PERFORMANCE DIFFERENCES ===")
    
    diff_count = 0
    for song in reports['wav']:
        if song == "OVERALL" or reports['wav'][song] is None:
            continue
            
        err_wav = reports['wav'][song]
        err_mp3 = reports['mp3'][song]
        err_ogg = reports['ogg'][song]
        
        # Check if the errors are exactly the same
        if not (err_wav == err_mp3 == err_ogg):
            diff_count += 1
            print(f"\n[ {song} ] had different performance based on format:")
            print(f"  WAV Error: {err_wav:.4f}")
            print(f"  MP3 Error: {err_mp3:.4f}")
            print(f"  OGG Error: {err_ogg:.4f}")
            
            # Find the best
            errors = {'WAV': err_wav, 'MP3': err_mp3, 'OGG': err_ogg}
            best_fmt = min(errors, key=errors.get)
            print(f"  -> Best format for this song: {best_fmt}")
            
    print(f"\nTotal songs where format changed the accuracy: {diff_count} / 50")

if __name__ == "__main__":
    main()
