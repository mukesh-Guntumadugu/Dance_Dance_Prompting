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
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpm", type=int, required=True, help="Exact BPM to generate")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    parser.add_argument("--csv_summary", type=str, default=None, help="Path to global CSV summary file")
    args = parser.parse_args()

    sr = 44100
    duration = 180.0 # Strictly 3 minutes
    total_samples = int(duration * sr)
    audio = np.zeros(total_samples)

    song_name = f"bpm_{args.bpm}"

    segments = [{
        "start": 0.0,
        "end": duration,
        "bpm": float(args.bpm)
    }]
    
    kick_high = random.uniform(120, 180)
    kick_low = random.uniform(40, 60)
    hat_decay = random.uniform(30, 60)
    
    kick = generate_kick_drum(sr, kick_high, kick_low)
    hihat = generate_hihat(sr, hat_decay)
    
    drop_probability = random.uniform(0.0, 0.1)
    syncopate_probability = random.uniform(0.0, 0.2)

    beat_timestamps = []
    onset_timestamps = []

    current_time = 0.0
    beat_index = 0
    
    beat_interval = 60.0 / float(args.bpm)
    
    while current_time < duration:
        beat_timestamps.append(current_time)
        is_kick = (beat_index % 2 == 0)
        instrument = kick if is_kick else hihat
        
        start_sample = int(current_time * sr)
        end_sample = start_sample + len(instrument)
        
        dropped = random.random() < drop_probability
        
        if not dropped and end_sample <= total_samples:
            audio[start_sample:end_sample] += instrument
            onset_timestamps.append(current_time)
            
        if random.random() < syncopate_probability:
            synco_time = current_time + (beat_interval / 2.0)
            if synco_time < duration:
                synco_start = int(synco_time * sr)
                synco_end = synco_start + len(hihat)
                if synco_end <= total_samples:
                    audio[synco_start:synco_end] += (hihat * 0.5)
                    onset_timestamps.append(synco_time)
                    
        current_time += beat_interval
        beat_index += 1

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    os.makedirs(args.output_dir, exist_ok=True)
    
    wav_path = os.path.join(args.output_dir, f"{song_name}.wav")
    sf.write(wav_path, audio, sr)
    
    ogg_path = os.path.join(args.output_dir, f"{song_name}.ogg")
    mp3_path = os.path.join(args.output_dir, f"{song_name}.mp3")
    
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "64k", ogg_path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    ground_truth = {
        "song_name": song_name,
        "duration": duration,
        "interval_type": 0,
        "segments": segments,
        "beats": beat_timestamps,
        "onsets": sorted(onset_timestamps)
    }
    
    json_path = os.path.join(args.output_dir, f"{song_name}_groundtruth.json")
    with open(json_path, 'w') as f:
        json.dump(ground_truth, f, indent=4)
        
    sm_path = os.path.join(args.output_dir, f"{song_name}.sm")
    with open(sm_path, 'w') as f:
        f.write(f"#TITLE:{song_name};\n")
        f.write(f"#BPMS:0.000={args.bpm:.3f};\n")
        
    if args.csv_summary:
        with open(args.csv_summary, 'a') as f:
            f.write(f"{song_name},{duration:.1f},BPM_Sweep,\"{args.bpm:.1f}\"\n")
        
    print(f"Generated {song_name} ({duration:.1f}s) -> .wav, .ogg, .mp3")

if __name__ == "__main__":
    main()
