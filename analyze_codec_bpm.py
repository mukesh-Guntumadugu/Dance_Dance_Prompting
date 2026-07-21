import argparse
import sys
import os
import torch
import soundfile as sf
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

import importlib.util
encodec_impl_path = os.path.join(os.path.dirname(__file__), "src/Neural Audio Codecs/EnCodecimplementation.py")
spec = importlib.util.spec_from_file_location("EnCodecimplementation", encodec_impl_path)
module = importlib.util.module_from_spec(spec)
sys.modules["EnCodecimplementation"] = module
spec.loader.exec_module(module)
AudioTokenizer = module.AudioTokenizer

from encodec.utils import convert_audio

def calculate_rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    error = original - reconstructed
    squared_error = error ** 2
    mse = torch.mean(squared_error)
    rmse = torch.sqrt(mse)
    return rmse.item()

def measure_codec_and_bpm(audio_path: str):
    print(f"========================================")
    print(f"MEASURING AUDIO CODEC RECONSTRUCTION LOSS AND BPM")
    print(f"File: {audio_path}")
    print(f"========================================")
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    # Using 1.5kbps (low quality) and 24kbps (high quality) for comparison
    bandwidths = [1.5, 24.0]
    
    print(f"Original Audio Analysis:")
    wav_np, sr = sf.read(audio_path)
    
    # Calculate Original BPM
    # converting to mono for librosa
    mono_np = wav_np if wav_np.ndim == 1 else wav_np.mean(axis=1)
    orig_tempo, _ = librosa.beat.beat_track(y=mono_np, sr=sr)
    orig_bpm = float(orig_tempo[0] if isinstance(orig_tempo, np.ndarray) else orig_tempo)
    print(f"  Original BPM: {orig_bpm:.2f}")

    results = []
    
    # Prepare plotting
    fig, axes = plt.subplots(len(bandwidths) + 1, 2, figsize=(15, 4 * (len(bandwidths) + 1)))
    
    # Plot original
    axes[0, 0].set_title("Original Waveform")
    librosa.display.waveshow(mono_np, sr=sr, ax=axes[0, 0])
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(mono_np)), ref=np.max)
    axes[0, 1].set_title("Original Spectrogram")
    librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=axes[0, 1])

    for i, bw in enumerate(bandwidths):
        print(f"\n--- Testing Bandwidth: {bw} kbps ---")
        tokenizer = AudioTokenizer(target_bandwidth=bw)
        
        original_wav = torch.from_numpy(wav_np).float()
        if original_wav.dim() == 1: 
            original_wav = original_wav.unsqueeze(0)
        else: 
            original_wav = original_wav.t()
        
        if sr != tokenizer.model.sample_rate:
             original_wav = convert_audio(original_wav.unsqueeze(0), sr, tokenizer.model.sample_rate, tokenizer.model.channels).squeeze(0)
        
        tokens = tokenizer.tokenize(audio_path)
        print(f"  Encoded frames: {tokens.shape[2]} | Codebooks: {tokens.shape[1]}")
        
        reconstructed_audio = tokenizer.decode(tokens).squeeze(0)
        
        # Audio metrics
        metrics = tokenizer.calculate_audio_metrics(original_wav, reconstructed_audio.cpu())
        
        # RMSE Alignment
        search_window = int(2.0 * tokenizer.model.sample_rate)
        tgt_mono = original_wav.mean(dim=0, keepdim=True) if original_wav.shape[0] > 1 else original_wav
        est_mono = reconstructed_audio.cpu().mean(dim=0, keepdim=True) if reconstructed_audio.shape[0] > 1 else reconstructed_audio.cpu()
        
        mid_idx = original_wav.shape[-1] // 2
        chunk_len = int(0.2 * tokenizer.model.sample_rate)
        start_idx = max(0, mid_idx - chunk_len // 2)
        end_idx = min(original_wav.shape[-1], mid_idx + chunk_len // 2)
        ref_chunk = tgt_mono[..., start_idx:end_idx]
        
        recon_start = max(0, start_idx - search_window)
        recon_end = min(reconstructed_audio.shape[-1], end_idx + search_window)
        query_chunk = est_mono[..., recon_start:recon_end]
        
        import torch.nn.functional as F
        kernel = ref_chunk.view(1, 1, -1)
        input_signal = query_chunk.view(1, 1, -1)
        out = F.conv1d(input_signal, kernel)
        best_idx = torch.argmax(out)
        offset = (recon_start + best_idx) - start_idx
        
        aligned_recon = reconstructed_audio.cpu()
        if offset > 0:
            aligned_recon = aligned_recon[..., offset:]
        elif offset < 0:
            aligned_recon = torch.cat([torch.zeros_like(aligned_recon[..., :int(-offset)]), aligned_recon], dim=-1)
            
        min_len = min(original_wav.shape[-1], aligned_recon.shape[-1])
        orig_trim = original_wav[..., :min_len]
        recon_trim = aligned_recon[..., :min_len]
        
        rmse = calculate_rmse(orig_trim, recon_trim)
        
        # Reconstructed BPM Calculation
        rec_np = reconstructed_audio.cpu().numpy()
        rec_mono_np = rec_np if rec_np.ndim == 1 else rec_np.mean(axis=0)
        rec_tempo, _ = librosa.beat.beat_track(y=rec_mono_np, sr=tokenizer.model.sample_rate)
        rec_bpm = float(rec_tempo[0] if isinstance(rec_tempo, np.ndarray) else rec_tempo)
        
        print(f"  Si-SNR: {metrics['Si-SNR']:.2f} dB")
        print(f"  RMSE: {rmse:.6f}")
        print(f"  Decoded BPM: {rec_bpm:.2f} (Diff: {abs(orig_bpm - rec_bpm):.2f})")
        
        # Plotting
        ax_idx = i + 1
        axes[ax_idx, 0].set_title(f"Decoded Waveform ({bw} kbps)")
        librosa.display.waveshow(rec_mono_np, sr=tokenizer.model.sample_rate, ax=axes[ax_idx, 0], color='orange')
        
        D_rec = librosa.amplitude_to_db(np.abs(librosa.stft(rec_mono_np)), ref=np.max)
        axes[ax_idx, 1].set_title(f"Decoded Spectrogram ({bw} kbps)")
        librosa.display.specshow(D_rec, y_axis='hz', x_axis='time', sr=tokenizer.model.sample_rate, ax=axes[ax_idx, 1])

        results.append({
            "Bandwidth": bw,
            "Si-SNR": metrics['Si-SNR'],
            "RMSE": rmse,
            "BPM_Orig": orig_bpm,
            "BPM_Decoded": rec_bpm
        })
        
    out_img = os.path.join(os.path.dirname(__file__), "outputs", "codec_bpm_comparison.png")
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_img)
    print(f"\nSaved visualization to {out_img}")

    print("\n============================================================")
    print("SUMMARY")
    print("============================================================")
    print(f"{'Bandwidth (kbps)':<20} | {'Si-SNR (dB) ↑':<15} | {'RMSE ↓':<15} | {'Orig BPM':<10} | {'Decoded BPM':<10}")
    print("-" * 80)
    for res in results:
        print(f"{res['Bandwidth']:<20} | {res['Si-SNR']:<15.2f} | {res['RMSE']:<15.6f} | {res['BPM_Orig']:<10.2f} | {res['BPM_Decoded']:<10.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    args = parser.parse_args()
    measure_codec_and_bpm(args.audio)
