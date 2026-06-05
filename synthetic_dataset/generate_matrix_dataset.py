#!/usr/bin/env python3
import os
import json
import argparse
import random
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

def synthesize_bpm_segment(bpm, duration, sr, current_time, kick, hihat, audio_buffer, total_samples):
    beat_interval = 60.0 / bpm
    beat_timestamps = []
    onset_timestamps = []
    
    # We want to fill `duration` seconds of audio exactly.
    segment_end_time = current_time + duration
    
    drop_probability = random.uniform(0.0, 0.1)
    syncopate_probability = random.uniform(0.0, 0.2)
    
    beat_index = 0
    while current_time < segment_end_time:
        beat_timestamps.append(current_time)
        is_kick = (beat_index % 2 == 0)
        instrument = kick if is_kick else hihat
        
        start_sample = int(current_time * sr)
        end_sample = start_sample + len(instrument)
        
        dropped = random.random() < drop_probability
        
        if not dropped and end_sample <= total_samples:
            audio_buffer[start_sample:end_sample] += instrument
            onset_timestamps.append(current_time)
            
        if random.random() < syncopate_probability:
            synco_time = current_time + (beat_interval / 2.0)
            if synco_time < segment_end_time:
                synco_start = int(synco_time * sr)
                synco_end = synco_start + len(hihat)
                if synco_end <= total_samples:
                    audio_buffer[synco_start:synco_end] += (hihat * 0.5)
                    onset_timestamps.append(synco_time)
                    
        current_time += beat_interval
        beat_index += 1
        
    return segment_end_time, beat_timestamps, onset_timestamps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_bpm", type=int, required=True, help="Base BPM to transition from")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save outputs")
    args = parser.parse_args()

    sr = 44100
    
    # Calculate exact total duration
    # 181 target BPMs (from 60 to 240)
    # Each pair is 20s of Base, 20s of Target = 40s per pair.
    # 181 * 40 = 7240 seconds.
    segment_duration = 20.0
    os.makedirs(args.output_dir, exist_ok=True)
    
    song_name = f"base_bpm_{args.base_bpm}"
    json_path = os.path.join(args.output_dir, f"{song_name}_groundtruth.json")
    if os.path.exists(json_path):
        print(f"Skipping {song_name}, already generated!")
        return

    targets = list(range(60, 241))
    
    total_duration = len(targets) * (segment_duration * 2)
    total_samples = int(total_duration * sr)
    
    print(f"Generating matrix for Base BPM {args.base_bpm} -> {len(targets)} targets. Total duration: {total_duration}s")
    
    audio = np.zeros(total_samples)

    song_name = f"base_bpm_{args.base_bpm}"
    
    kick_high = random.uniform(120, 180)
    kick_low = random.uniform(40, 60)
    hat_decay = random.uniform(30, 60)
    
    kick = generate_kick_drum(sr, kick_high, kick_low)
    hihat = generate_hihat(sr, hat_decay)
    
    current_time = 0.0
    segments_json = []
    all_beats = []
    all_onsets = []

    for target_bpm in targets:
        # Part 1: Base BPM for 20s
        segments_json.append({"start": current_time, "end": current_time + segment_duration, "bpm": float(args.base_bpm)})
        current_time, beats, onsets = synthesize_bpm_segment(
            args.base_bpm, segment_duration, sr, current_time, kick, hihat, audio, total_samples
        )
        all_beats.extend(beats)
        all_onsets.extend(onsets)
        
        # Part 2: Target BPM for 20s
        segments_json.append({"start": current_time, "end": current_time + segment_duration, "bpm": float(target_bpm)})
        current_time, beats, onsets = synthesize_bpm_segment(
            target_bpm, segment_duration, sr, current_time, kick, hihat, audio, total_samples
        )
        all_beats.extend(beats)
        all_onsets.extend(onsets)

    # Normalize audio
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val * 0.9

    os.makedirs(args.output_dir, exist_ok=True)
    
    wav_path = os.path.join(args.output_dir, f"{song_name}.wav")
    print(f"Saving {wav_path}...")
    sf.write(wav_path, audio, sr)
    
    # Free the 1.2GB audio array before JSON serialization to prevent OOM
    del audio

    ground_truth = {
        "song_name": song_name,
        "duration": total_duration,
        "segments": segments_json,
        "beats": all_beats,
        "onsets": sorted(all_onsets)
    }
    
    json_path = os.path.join(args.output_dir, f"{song_name}_groundtruth.json")
    with open(json_path, 'w') as f:
        json.dump(ground_truth, f, indent=4)
        
    print(f"Finished {song_name}!")

if __name__ == "__main__":
    main()
