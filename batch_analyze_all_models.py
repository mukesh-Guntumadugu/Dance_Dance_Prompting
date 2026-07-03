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

# Try to load EnCodec for MuMu-LLaMA
try:
    encodec_impl_path = os.path.join(os.path.dirname(__file__), "src/Neural Audio Codecs/EnCodecimplementation.py")
    spec = importlib.util.spec_from_file_location("EnCodecimplementation", encodec_impl_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["EnCodecimplementation"] = module
    spec.loader.exec_module(module)
    AudioTokenizer = module.AudioTokenizer
    from encodec.utils import convert_audio
    HAS_ENCODEC = True
except ImportError as e:
    HAS_ENCODEC = False
    print(f"Warning: Could not load EnCodec: {e}")

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

def run_large_scale_batch():
    bpms = range(60, 241, 20)
    results = []
    
    print("=== STARTING LARGE-SCALE MULTI-MODEL BATCH ANALYSIS ===")
    
    # Initialize codecs
    if HAS_ENCODEC:
        print("Loading EnCodec models for MuMu-LLaMA...")
        encodec_1_5 = AudioTokenizer(target_bandwidth=1.5)
        encodec_24 = AudioTokenizer(target_bandwidth=24.0)
    
    # Model configuration definitions
    models_to_test = [
        {
            "name": "MuMu-LLaMA (Low Res)",
            "encoder": "EnCodec (Discrete RVQ)",
            "frames": "75 frames/sec",
            "quantization": "1.5 kbps (2 Codebook Layers)",
            "supports_decode": True,
            "tokenizer": encodec_1_5 if HAS_ENCODEC else None
        },
        {
            "name": "MuMu-LLaMA (High Res)",
            "encoder": "EnCodec (Discrete RVQ)",
            "frames": "75 frames/sec",
            "quantization": "24 kbps (32 Codebook Layers)",
            "supports_decode": True,
            "tokenizer": encodec_24 if HAS_ENCODEC else None
        },
        {
            "name": "DeepResonance",
            "encoder": "ImageBind (Continuous)",
            "frames": "Variable (Continuous Embeddings)",
            "quantization": "None (Dense fp16 Vectors)",
            "supports_decode": False,
            "tokenizer": None
        },
        {
            "name": "Flamingo",
            "encoder": "AudioFlamingo Backbone",
            "frames": "Continuous Latents",
            "quantization": "None (Dense fp16 Vectors)",
            "supports_decode": False,
            "tokenizer": None
        },
        {
            "name": "Macaw / Qwen",
            "encoder": "Whisper (Continuous)",
            "frames": "50 frames/sec",
            "quantization": "None (Dense fp16 Vectors)",
            "supports_decode": False,
            "tokenizer": None
        }
    ]

    for target_bpm in bpms:
        print(f"\nProcessing BPM: {target_bpm}")
        audio_path = generate_click_track(target_bpm)
        wav_np, sr = sf.read(audio_path)
        mono_np = wav_np if wav_np.ndim == 1 else wav_np.mean(axis=1)
        
        orig_tempo, _ = librosa.beat.beat_track(y=mono_np, sr=sr)
        orig_bpm_est = float(orig_tempo[0] if isinstance(orig_tempo, np.ndarray) else orig_tempo)
        
        for model in models_to_test:
            result_row = {
                "Model": model["name"],
                "Encoder_Type": model["encoder"],
                "Frames_Per_Sec": model["frames"],
                "Quantization_Capacity": model["quantization"],
                "Target_BPM": target_bpm,
                "Orig_Librosa_BPM": round(orig_bpm_est, 2),
                "Encoded_Data_Size": "N/A",
                "Si-SNR_Loss": "N/A",
                "RMSE_Loss": "N/A",
                "Decoded_Librosa_BPM": "N/A",
                "Tempo_Diff": "N/A"
            }
            
            if model["supports_decode"] and model["tokenizer"] is not None:
                tokenizer = model["tokenizer"]
                original_wav = torch.from_numpy(wav_np).float().unsqueeze(0)
                if sr != tokenizer.model.sample_rate:
                    original_wav = convert_audio(original_wav.unsqueeze(0), sr, tokenizer.model.sample_rate, tokenizer.model.channels).squeeze(0)
                
                # ENCODE
                tokens = tokenizer.tokenize(audio_path)
                result_row["Encoded_Data_Size"] = f"{tokens.shape[1]} layers x {tokens.shape[2]} tokens"
                
                # DECODE
                reconstructed_audio = tokenizer.decode(tokens).squeeze(0)
                metrics = tokenizer.calculate_audio_metrics(original_wav, reconstructed_audio.cpu())
                
                # RMSE Alignment
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
                
                rec_np = reconstructed_audio.cpu().numpy()
                rec_mono = rec_np if rec_np.ndim == 1 else rec_np.mean(axis=0)
                rec_tempo, _ = librosa.beat.beat_track(y=rec_mono, sr=tokenizer.model.sample_rate)
                rec_bpm_est = float(rec_tempo[0] if isinstance(rec_tempo, np.ndarray) else rec_tempo)
                
                result_row["Si-SNR_Loss"] = round(metrics['Si-SNR'], 2)
                result_row["RMSE_Loss"] = round(rmse, 6)
                result_row["Decoded_Librosa_BPM"] = round(rec_bpm_est, 2)
                result_row["Tempo_Diff"] = round(abs(orig_bpm_est - rec_bpm_est), 2)
            else:
                result_row["Encoded_Data_Size"] = "Dense (768/1024 dims)"
                result_row["Si-SNR_Loss"] = "N/A (Continuous)"
                result_row["RMSE_Loss"] = "N/A (Continuous)"
                result_row["Decoded_Librosa_BPM"] = "N/A (No Decoder)"
                result_row["Tempo_Diff"] = "N/A (No Decoder)"
                
            results.append(result_row)
            
    df = pd.DataFrame(results)
    out_csv = os.path.join(os.path.dirname(__file__), "outputs", "large_scale_model_codec_analysis.csv")
    df.to_csv(out_csv, index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY PREVIEW (First few entries)")
    print("="*80)
    print(df.head(10).to_string())
    print("\nSaved full multi-model results to:", out_csv)

if __name__ == "__main__":
    run_large_scale_batch()
