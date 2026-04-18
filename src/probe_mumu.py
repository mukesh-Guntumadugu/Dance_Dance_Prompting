import os
import argparse
import librosa
import torchaudio
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings

# Suppress annoying warnings
warnings.filterwarnings('ignore')

import mumu_measure_interface

def plot_for_audio(file_path, output_dir):
    print(f"\n========================================")
    print(f"PROBING MUMU-LLAMA: {os.path.basename(file_path)}")
    print(f"========================================")
    
    y, sr = librosa.load(file_path, sr=None)
    duration = librosa.get_duration(y=y, sr=sr)
    
    print("1. Extracting mathematical ground truths (Librosa)...")
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(onset_env, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    model, tokenizer = mumu_measure_interface._mumu_model, mumu_measure_interface._mumu_tokenizer
    
    print("2. Pushing audio through MuMu MERT backbone...")
    features = None
    with torch.no_grad():
        try:
            # Load the exact MuMu audio tensor.
            # CRITICAL: MERT expects 24kHz audio as its absolute requirement.
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != 24000:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=24000)
            
            # MERT usually takes [Batch, Seq] shape and mono
            audio_tensor = torch.mean(waveform, 0).unsqueeze(0).cuda()
            
            print("   -> Directly intercepting MERT `encode_audio` pipeline...")
            # Run the audio through MERT, downsampling projections, and into the MuMu dense projection space
            features = model.encode_audio(audio_tensor)
            
            if features is None:
                raise ValueError("Model encode_audio returned None")
                
            # If it's a tuple (which some LLaMA adapters return as (features, attention_masks)), grab the tensor
            if isinstance(features, tuple):
                features = features[0]
                
            features = features.cpu().numpy()
            
        except Exception as e:
            print(f"   [!] MuMu Neural Extraction Failed: {e}")
            import traceback
            traceback.print_exc()
            return
            
    print(f"   -> Extracted latent tokens with dimensions: {features.shape}.")
    
    print("3. Compressing MuMu latents via PCA...")
    pca = PCA(n_components=1)
    
    if len(features.shape) > 2:
        features = features.reshape(-1, features.shape[-1])
        
    mumu_1d = pca.fit_transform(features).flatten()
    
    if abs(min(mumu_1d)) > abs(max(mumu_1d)):
        mumu_1d = -mumu_1d
        
    mumu_1d = (mumu_1d - mumu_1d.min()) / (mumu_1d.max() - mumu_1d.min() + 1e-8)
    onset_norm = (onset_env - onset_env.min()) / (onset_env.max() - onset_env.min() + 1e-8)
    
    mumu_times = np.linspace(0, duration, len(mumu_1d))
    
    print("4. Generating visual correlation chart...")
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    out_file = os.path.join(output_dir, f"{base_name}_mumu_onset_probe.png")
    
    plt.figure(figsize=(18, 6))
    plt.plot(times, onset_norm, label="Librosa Mathematical Onset", color="blue", alpha=0.35, linewidth=2)
    for i, b in enumerate(beat_times):
        plt.axvline(x=b, color='green', alpha=0.3, linestyle='--', linewidth=1, label="Mathematical Drum Beats" if i == 0 else "")
        
    plt.plot(mumu_times, mumu_1d, label="MuMu (MERT) Activation (1D PCA)", color="magenta", linewidth=2.5)
    
    plt.title(f"MuMu-LLaMA (MERT) Understanding vs Physical Tempo\nTarget: {base_name}", fontsize=14)
    plt.xlabel("Time (Seconds)", fontsize=11)
    plt.ylabel("Activation Spike Intensity (Normalized)", fontsize=11)
    plt.legend(loc="upper right")
    plt.grid(alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_file, dpi=180)
    plt.close()
    
    print(f"✅ MuMu-LLaMA Chart exported: {out_file}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True, help="Directory containing audio")
    parser.add_argument('--output_dir', type=str, default="results_mumu_probe_fraxtil", help="Output directory")
    args = parser.parse_args()
    
    args.target_dir = os.path.abspath(args.target_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n--- Initializing MuMu-LLaMA Audio Engine ---")
    mumu_measure_interface.initialize_mumu_model()
    
    supported_exts = ['.ogg', '.mp3', '.wav']
    files = []
    for root, _, filenames in os.walk(args.target_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in supported_exts:
                files.append(os.path.join(root, f))
                
    if not files:
        print(f"❌ No audio files found.")
        return
        
    for fp in files:
        plot_for_audio(fp, args.output_dir)

if __name__ == "__main__":
    main()
