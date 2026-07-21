import argparse
import sys
import os
import torch
import soundfile as sf
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Import AudioTokenizer from the EnCodec implementation
import importlib.util
encodec_impl_path = os.path.join(os.path.dirname(__file__), "src/Neural Audio Codecs/EnCodecimplementation.py")
spec = importlib.util.spec_from_file_location("EnCodecimplementation", encodec_impl_path)
module = importlib.util.module_from_spec(spec)
sys.modules["EnCodecimplementation"] = module
spec.loader.exec_module(module)
AudioTokenizer = module.AudioTokenizer

from encodec.utils import convert_audio

def calculate_rmse(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """Calculate the Root Mean Square Error (RMSE) between two aligned waveforms."""
    # Compute error
    error = original - reconstructed
    # Square the error
    squared_error = error ** 2
    # Mean of squared error
    mse = torch.mean(squared_error)
    # Root of mean squared error
    rmse = torch.sqrt(mse)
    return rmse.item()

def measure_loss(audio_path: str):
    print(f"========================================")
    print(f"MEASURING AUDIO CODEC RECONSTRUCTION LOSS")
    print(f"File: {audio_path}")
    print(f"========================================")
    
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    # Bandwidths supported by EnCodec 24kHz model
    bandwidths = [1.5, 3.0, 6.0, 12.0, 24.0]
    
    print(f"\nWe will test EnCodec across its supported bitrates (kbps): {bandwidths}")
    print(f"This shows how much data is lost during encoding at different compression levels.\n")

    results = []

    for bw in bandwidths:
        print(f"--- Testing Bandwidth: {bw} kbps ---")
        tokenizer = AudioTokenizer(target_bandwidth=bw)
        
        # Load original audio manually for comparison
        wav_np, sr = sf.read(audio_path)
        original_wav = torch.from_numpy(wav_np).float()
        if original_wav.dim() == 1: 
            original_wav = original_wav.unsqueeze(0)
        else: 
            original_wav = original_wav.t()
        
        # Resample original to match tokenizer sample rate (usually 24kHz)
        if sr != tokenizer.model.sample_rate:
             original_wav = convert_audio(original_wav.unsqueeze(0), sr, tokenizer.model.sample_rate, tokenizer.model.channels).squeeze(0)
        
        # 1. Encode
        tokens = tokenizer.tokenize(audio_path)
        print(f"  Encoded to {tokens.shape[1]} codebooks. Total tokens: {tokens.shape[1] * tokens.shape[2]}")
        
        # 2. Decode
        reconstructed_audio = tokenizer.decode(tokens)
        reconstructed_audio = reconstructed_audio.squeeze(0) # remove batch
        
        # 3. Use EnCodec implementation's builtin alignment and Si-SNR/Corr computation
        metrics = tokenizer.calculate_audio_metrics(original_wav, reconstructed_audio.cpu())
        
        # 4. We also want RMSE. We need to apply the same alignment.
        # Repeating the alignment logic here to compute exact RMSE
        # (calculate_audio_metrics shifts the arrays but doesn't return the aligned arrays, so we do it briefly)
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
        metrics["RMSE"] = rmse
        
        print(f"  Si-SNR: {metrics['Si-SNR']:.2f} dB")
        print(f"  Correlation: {metrics['Correlation']:.4f}")
        print(f"  RMSE: {rmse:.6f}")
        
        # Save output for this bandwidth
        out_dir = os.path.join(os.path.dirname(__file__), "outputs")
        os.makedirs(out_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        out_wav = os.path.join(out_dir, f"{base_name}_reconstructed_{bw}kbps.wav")
        sf.write(out_wav, reconstructed_audio.cpu().numpy().T, tokenizer.model.sample_rate)
        print(f"  Saved reconstructed audio to: {out_wav}\n")
        
        results.append({
            "Bandwidth": bw,
            "Si-SNR": metrics['Si-SNR'],
            "Correlation": metrics['Correlation'],
            "RMSE": rmse
        })
        
    print("========================================")
    print("SUMMARY OF RECONSTRUCTION LOSS")
    print("========================================")
    print(f"{'Bandwidth':<15} | {'Si-SNR (dB) ↑':<15} | {'Correlation ↑':<15} | {'RMSE ↓':<15}")
    print("-" * 65)
    for res in results:
        print(f"{res['Bandwidth']:<15} | {res['Si-SNR']:<15.2f} | {res['Correlation']:<15.4f} | {res['RMSE']:<15.6f}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Encoder Reconstruction Loss")
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file")
    args = parser.parse_args()
    measure_loss(args.audio)
