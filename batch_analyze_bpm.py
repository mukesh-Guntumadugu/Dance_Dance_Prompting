import os
import sys
import torch
import soundfile as sf
import numpy as np
import librosa
import warnings
import pandas as pd

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
    return torch.sqrt(torch.mean(error ** 2)).item()

def generate_click_track(bpm, duration=10, sr=24000):
    beat_interval = 60.0 / bpm
    times = np.arange(0, duration, beat_interval)
    y = librosa.clicks(times=times, sr=sr, length=int(sr * duration), click_freq=1000.0, click_duration=0.1)
    noise = np.random.normal(0, 0.005, len(y))
    y = y + noise
    
    out_dir = os.path.join(os.path.dirname(__file__), "batch_audio")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"test_{bpm}bpm.wav")
    sf.write(filename, y, sr)
    return filename

def run_batch_experiment():
    bpms = range(60, 241, 20)  # 60, 80, 100... 240
    bandwidths = [1.5, 24.0]
    
    results = []
    
    print("Initializing EnCodec models...")
    tokenizers = {bw: AudioTokenizer(target_bandwidth=bw) for bw in bandwidths}
    
    print(f"{'Target BPM':<12} | {'BW (kbps)':<10} | {'Orig Librosa':<15} | {'Decoded Librosa':<18} | {'Si-SNR (dB)':<12} | {'RMSE':<10}")
    print("-" * 85)
    
    for target_bpm in bpms:
        audio_path = generate_click_track(target_bpm)
        wav_np, sr = sf.read(audio_path)
        mono_np = wav_np if wav_np.ndim == 1 else wav_np.mean(axis=1)
        
        # Original librosa BPM (we supply the exact starting BPM as prior if we want, 
        # but let's see what librosa predicts blindly to answer the user's question)
        orig_tempo, _ = librosa.beat.beat_track(y=mono_np, sr=sr)
        orig_bpm_est = float(orig_tempo[0] if isinstance(orig_tempo, np.ndarray) else orig_tempo)
        
        for bw in bandwidths:
            tokenizer = tokenizers[bw]
            
            original_wav = torch.from_numpy(wav_np).float().unsqueeze(0)
            if sr != tokenizer.model.sample_rate:
                original_wav = convert_audio(original_wav.unsqueeze(0), sr, tokenizer.model.sample_rate, tokenizer.model.channels).squeeze(0)
            
            tokens = tokenizer.tokenize(audio_path)
            reconstructed_audio = tokenizer.decode(tokens).squeeze(0)
            
            metrics = tokenizer.calculate_audio_metrics(original_wav, reconstructed_audio.cpu())
            
            # Simplified RMSE alignment
            search_window = int(2.0 * tokenizer.model.sample_rate)
            tgt_mono = original_wav.mean(dim=0, keepdim=True)
            est_mono = reconstructed_audio.cpu().mean(dim=0, keepdim=True)
            mid_idx = original_wav.shape[-1] // 2
            chunk_len = int(0.2 * tokenizer.model.sample_rate)
            start_idx = max(0, mid_idx - chunk_len // 2)
            end_idx = min(original_wav.shape[-1], mid_idx + chunk_len // 2)
            ref_chunk = tgt_mono[..., start_idx:end_idx]
            recon_start = max(0, start_idx - search_window)
            recon_end = min(reconstructed_audio.shape[-1], end_idx + search_window)
            query_chunk = est_mono[..., recon_start:recon_end]
            
            import torch.nn.functional as F
            out = F.conv1d(query_chunk.view(1, 1, -1), ref_chunk.view(1, 1, -1))
            offset = (recon_start + torch.argmax(out)) - start_idx
            
            aligned_recon = reconstructed_audio.cpu()
            if offset > 0: aligned_recon = aligned_recon[..., offset:]
            elif offset < 0: aligned_recon = torch.cat([torch.zeros_like(aligned_recon[..., :int(-offset)]), aligned_recon], dim=-1)
            min_len = min(original_wav.shape[-1], aligned_recon.shape[-1])
            rmse = calculate_rmse(original_wav[..., :min_len], aligned_recon[..., :min_len])
            
            # Decoded Librosa BPM
            rec_np = reconstructed_audio.cpu().numpy()
            rec_mono = rec_np if rec_np.ndim == 1 else rec_np.mean(axis=0)
            rec_tempo, _ = librosa.beat.beat_track(y=rec_mono, sr=tokenizer.model.sample_rate)
            rec_bpm_est = float(rec_tempo[0] if isinstance(rec_tempo, np.ndarray) else rec_tempo)
            
            print(f"{target_bpm:<12} | {bw:<10.1f} | {orig_bpm_est:<15.2f} | {rec_bpm_est:<18.2f} | {metrics['Si-SNR']:<12.2f} | {rmse:<10.6f}")
            
            results.append({
                "Target_BPM": target_bpm,
                "Bandwidth": bw,
                "Orig_Librosa_BPM": orig_bpm_est,
                "Decoded_Librosa_BPM": rec_bpm_est,
                "Si-SNR": metrics['Si-SNR'],
                "RMSE": rmse
            })
            
    df = pd.DataFrame(results)
    out_csv = os.path.join(os.path.dirname(__file__), "outputs", "batch_bpm_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved full results to {out_csv}")

if __name__ == "__main__":
    run_batch_experiment()
