#!/usr/bin/env python3
import os
import json
import argparse
import random
import subprocess
import numpy as np
import soundfile as sf

def generate_kick_drum(sr, base_high=150, base_low=50, duration=0.2):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    freqs = np.linspace(base_high, base_low, len(t))
    phases = np.cumsum(freqs * 2 * np.pi / sr)
    waveform = np.sin(phases)
    envelope = np.exp(-15 * t)
    return waveform * envelope

def generate_hihat(sr, decay=40, duration=0.1):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    noise = np.random.uniform(-1, 1, len(t))
    envelope = np.exp(-decay * t)
    return noise * envelope

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic audio dataset V2")
    parser.add_argument("--song_name", type=str, required=True, help="Base name (e.g. song_1)")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--interval", type=float, required=True, help="0=constant, -1=random >5s, N=change every N seconds")
    parser.add_argument("--csv_summary", type=str, default=None, help="Path to global CSV summary file")
    parser.add_argument("--category", type=str, default="Unknown", help="Category name for CSV")
    args = parser.parse_args()

    sr = 44100
    
    # 1. Randomize Duration between 3 and 4 minutes (180 to 240 seconds)
    duration = random.uniform(180.0, 240.0)
    total_samples = int(duration * sr)
    audio = np.zeros(total_samples)

    # 2. Build BPM Segments
    segments = []
    current_time = 0.0
    
    while current_time < duration:
        if args.interval == 0:
            block_length = duration # Constant
        elif args.interval == -1:
            block_length = random.uniform(5.1, 25.0) # Random > 5s
        else:
            block_length = args.interval # Fixed interval
            
        block_end = min(current_time + block_length, duration)
        
        # Pick a random BPM between 80 and 170
        bpm = random.uniform(80.0, 170.0)
        
        segments.append({
            "start": current_time,
            "end": block_end,
            "bpm": bpm
        })
        current_time = block_end

    # 3. Randomize Drum synthesis parameters for this specific song
    kick_high = random.uniform(120, 180)
    kick_low = random.uniform(40, 60)
    hat_decay = random.uniform(30, 60)
    
    kick = generate_kick_drum(sr, kick_high, kick_low)
    hihat = generate_hihat(sr, hat_decay)
    
    # We will sometimes drop hits or add syncopation to prevent overfitting
    drop_probability = random.uniform(0.0, 0.1)
    syncopate_probability = random.uniform(0.0, 0.2)

    beat_timestamps = []
    onset_timestamps = []

    # 4. Generate Audio
    current_time = 0.0
    beat_index = 0
    
    for seg in segments:
        bpm = seg["bpm"]
        beat_interval = 60.0 / bpm
        
        # Fill this segment with beats
        while current_time < seg["end"]:
            beat_timestamps.append(current_time)
            
            # Place instruments (Onset logic)
            # Basic 4/4 feel: kick on 0 and 2, hat on 1 and 3 (relative to 4 beats)
            is_kick = (beat_index % 2 == 0)
            instrument = kick if is_kick else hihat
            
            start_sample = int(current_time * sr)
            end_sample = start_sample + len(instrument)
            
            # Apply occasional drop (don't play the note)
            dropped = random.random() < drop_probability
            
            if not dropped and end_sample <= total_samples:
                audio[start_sample:end_sample] += instrument
                onset_timestamps.append(current_time)
                
            # Apply occasional syncopation (play a hihat exactly halfway to next beat)
            if random.random() < syncopate_probability:
                synco_time = current_time + (beat_interval / 2.0)
                if synco_time < seg["end"]:
                    synco_start = int(synco_time * sr)
                    synco_end = synco_start + len(hihat)
                    if synco_end <= total_samples:
                        # Quieter syncopation
                        audio[synco_start:synco_end] += (hihat * 0.5)
                        onset_timestamps.append(synco_time)
                        
            current_time += beat_interval
            beat_index += 1

    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 5. Export WAV
    wav_path = os.path.join(args.output_dir, f"{args.song_name}.wav")
    sf.write(wav_path, audio, sr)
    
    # 6. Convert to OGG and MP3 using FFmpeg
    ogg_path = os.path.join(args.output_dir, f"{args.song_name}.ogg")
    mp3_path = os.path.join(args.output_dir, f"{args.song_name}.mp3")
    
    # Suppress ffmpeg output to keep console clean. Using libopus because libvorbis is missing on this Mac
    # and soundfile's native Vorbis encoder causes a segmentation fault on Apple Silicon.
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", ogg_path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 7. Export Ground Truth JSON
    ground_truth = {
        "song_name": args.song_name,
        "duration": duration,
        "interval_type": args.interval,
        "segments": segments,
        "beats": beat_timestamps,
        "onsets": sorted(onset_timestamps)
    }
    
    json_path = os.path.join(args.output_dir, f"{args.song_name}_groundtruth.json")
    with open(json_path, 'w') as f:
        json.dump(ground_truth, f, indent=4)
        
    # 8. Export mock .sm file (Using the first segment's BPM, though not ideal for multi-bpm, 
    # the JSON is the true ground truth. We'll write all BPMS to the .sm file properly.)
    sm_path = os.path.join(args.output_dir, f"{args.song_name}.sm")
    with open(sm_path, 'w') as f:
        f.write(f"#TITLE:{args.song_name};\n")
        bpm_string = ",".join([f"{(seg['start'] / (60.0/seg['bpm'])):.3f}={seg['bpm']:.3f}" for seg in segments])
        f.write(f"#BPMS:{bpm_string};\n")
        
    # 9. Append to summary CSV
    if args.csv_summary:
        bpm_seq_str = " -> ".join([f"{seg['bpm']:.1f}" for seg in segments])
        with open(args.csv_summary, 'a') as f:
            f.write(f"{args.song_name},{duration:.1f},{args.category},\"{bpm_seq_str}\"\n")
        
    print(f"Generated {args.song_name} ({duration:.1f}s) -> .wav, .ogg, .mp3, .json, .sm")

if __name__ == "__main__":
    main()
