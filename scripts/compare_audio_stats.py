#!/usr/bin/env python3
"""
compare_audio_stats.py
A script to extract and compare the average BPM, time duration, and mathematical onset density
between a baseline dataset (e.g., 1000 copyright-free songs) and the StepMania datasets.
"""

import os
import glob
import librosa
import numpy as np
import argparse
from multiprocessing import Pool
import pandas as pd

def analyze_audio(file_path):
    """Extracts duration, BPM, and mathematical onset density from a single audio file."""
    try:
        # Load audio (downsample to 22050 for faster processing)
        y, sr = librosa.load(file_path, sr=22050)
        duration = librosa.get_duration(y=y, sr=sr)
        
        if duration <= 0:
            return None
            
        # Calculate Onset Envelope (percussive/melodic hits)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Detect exact onset frames
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        num_onsets = len(onsets)
        onsets_per_sec = num_onsets / duration
        
        # Calculate Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo
        
        return {
            'duration': duration,
            'onsets_per_sec': onsets_per_sec,
            'bpm': bpm
        }
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
        return None

def process_directory(directory_path, name):
    print(f"\nScanning directory: {directory_path} ...")
    
    # Recursively find all audio files
    audio_files = []
    for ext in ['*.mp3', '*.ogg', '*.wav']:
        audio_files.extend(glob.glob(os.path.join(directory_path, '**', ext), recursive=True))
        
    if not audio_files:
        print(f"No audio files found in {directory_path}!")
        return None
        
    print(f"Found {len(audio_files)} audio files. Extracting librosa features (this may take a while)...")
    
    with Pool(os.cpu_count()) as pool:
        results = pool.map(analyze_audio, audio_files)
        
    valid_results = [r for r in results if r is not None]
    if not valid_results:
        return None
        
    df = pd.DataFrame(valid_results)
    
    return {
        'Dataset': name,
        'Num Songs': len(valid_results),
        'Avg Duration (s)': df['duration'].mean(),
        'Avg BPM': df['bpm'].mean(),
        'Avg Onsets / Sec': df['onsets_per_sec'].mean()
    }

def main():
    parser = argparse.ArgumentParser(description='Audio Statistics Comparer')
    parser.add_argument('--stepmania_dir', type=str, default='/data/mg546924/llm_beatmap_generator/src/musicForBeatmap', help='Path to StepMania songs')
    parser.add_argument('--baseline_dir', type=str, required=True, help='Path to 1000 Copyright Free Songs')
    parser.add_argument('--output', type=str, default='audio_comparison_table.md', help='Output markdown file')
    
    args = parser.parse_args()
    
    print("==================================================")
    print("🎶 Audio Feature Comparison: StepMania vs Baseline")
    print("==================================================")
    
    stats = []
    
    # 1. Process StepMania Dataset
    sm_stats = process_directory(args.stepmania_dir, "StepMania Dataset (Fraxtil/Mixed)")
    if sm_stats:
        stats.append(sm_stats)
        
    # 2. Process Baseline Copyright-Free Dataset
    baseline_stats = process_directory(args.baseline_dir, "Copyright-Free Baseline (1000 Songs)")
    if baseline_stats:
        stats.append(baseline_stats)
        
    if stats:
        df_out = pd.DataFrame(stats)
        
        # Round the values for clean presentation
        df_out['Avg Duration (s)'] = df_out['Avg Duration (s)'].round(1)
        df_out['Avg BPM'] = df_out['Avg BPM'].round(1)
        df_out['Avg Onsets / Sec'] = df_out['Avg Onsets / Sec'].round(2)
        
        # Convert Duration to Min:Sec string format
        df_out['Avg Duration'] = df_out['Avg Duration (s)'].apply(lambda x: f"{int(x // 60)}m {int(x % 60)}s")
        df_out = df_out.drop(columns=['Avg Duration (s)'])
        
        # Reorder columns
        df_out = df_out[['Dataset', 'Num Songs', 'Avg Duration', 'Avg BPM', 'Avg Onsets / Sec']]
        
        # Print to console
        print("\n\n📊 Final Results:")
        print(df_out.to_markdown(index=False))
        
        # Save to file
        with open(args.output, "w") as f:
            f.write("# Audio Feature Comparison\n\n")
            f.write(df_out.to_markdown(index=False))
            f.write("\n\n*Note: Onsets per second refers to raw mathematical percussive hits detected via librosa.*")
            
        print(f"\n✅ Saved comparison table to {args.output}")

if __name__ == "__main__":
    main()
