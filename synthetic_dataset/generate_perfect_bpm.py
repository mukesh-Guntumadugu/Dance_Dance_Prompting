#!/usr/bin/env python3
import os
import argparse
import random
import numpy as np
import wave
import struct

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
    parser.add_argument("--duration", type=float, default=60.0, help="Duration of the song in seconds")
    args = parser.parse_args()

    # Fixed seed for perfectly identical instruments and syncopation patterns
    random.seed(42)
    np.random.seed(42)

    sr = 44100
    total_samples = int(args.duration * sr)
    audio = np.zeros(total_samples)

    song_name = f"linear_probing_{args.bpm}_bpm_stable"

    kick_high = random.uniform(120, 180)
    kick_low = random.uniform(40, 60)
    hat_decay = random.uniform(30, 60)
    
    kick = generate_kick_drum(sr, kick_high, kick_low)
    hihat = generate_hihat(sr, hat_decay)
    
    drop_probability = random.uniform(0.0, 0.1)
    syncopate_probability = random.uniform(0.0, 0.2)

    current_time = 0.0
    beat_index = 0
    beat_interval = 60.0 / float(args.bpm)
    
    while current_time < args.duration:
        is_kick = (beat_index % 2 == 0)
        instrument = kick if is_kick else hihat
        
        start_sample = int(current_time * sr)
        end_sample = start_sample + len(instrument)
        
        dropped = random.random() < drop_probability
        if not dropped and end_sample <= total_samples:
            audio[start_sample:end_sample] += instrument
            
        if random.random() < syncopate_probability:
            synco_time = current_time + (beat_interval / 2.0)
            if synco_time < args.duration:
                synco_start = int(synco_time * sr)
                synco_end = synco_start + len(hihat)
                if synco_end <= total_samples:
                    audio[synco_start:synco_end] += (hihat * 0.5)
                    
        current_time += beat_interval
        beat_index += 1

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    os.makedirs(args.output_dir, exist_ok=True)
    wav_path = os.path.join(args.output_dir, f"{song_name}.wav")
    
    # Save using built-in wave module to remove soundfile dependency
    with wave.open(wav_path, 'w') as f:
        f.setnchannels(1)
        f.setsampwidth(2) # 16-bit audio
        f.setframerate(sr)
        audio_int16 = np.int16(audio * 32767)
        f.writeframes(audio_int16.tobytes())
    
    # Save the file path to README.md
    readme_path = os.path.join(args.output_dir, "README.md")
    with open(readme_path, "a") as f:
        f.write(f"- `{wav_path}`\n")
        
    print(f"Generated {song_name} ({args.duration:.1f}s)")

if __name__ == "__main__":
    main()
