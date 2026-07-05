#!/usr/bin/env python3
import os
import torch
import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformers import WhisperFeatureExtractor, WhisperModel
from tqdm import tqdm

def main():
    # Settings
    DATASET_DIR = "linear_probing_dataset"
    MODEL_ID = "openai/whisper-large-v2"
    OUTPUT_CSV = "linear_probing_correlations.csv"
    OUTPUT_PLOT = "linear_probing_top_dims.png"
    
    print(f"Loading {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    model = WhisperModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    
    bpms = []
    embeddings = []
    
    # Process 60 to 240 BPM
    print("Extracting Whisper embeddings for each file...")
    for bpm in tqdm(range(60, 241)):
        wav_path = os.path.join(DATASET_DIR, f"linear_probing_{bpm}_bpm_stable.wav")
        
        if not os.path.exists(wav_path):
            print(f"Warning: {wav_path} not found. Skipping.")
            continue
            
        # Whisper processes up to 30 seconds of audio at 16kHz
        # We'll load exactly 30s to get a robust embedding
        audio, sr = librosa.load(wav_path, sr=16000, duration=30.0)
        
        # Prepare inputs
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        # Get encoder hidden states
        with torch.no_grad():
            encoder_outputs = model.encoder(input_features)
            # Shape: [1, seq_len, hidden_size] (hidden_size = 1024 for large-v2)
            hidden_states = encoder_outputs.last_hidden_state
            
            # Mean pool across the time dimension to get a single 1024-d vector for this track
            pooled_vector = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            
        bpms.append(bpm)
        embeddings.append(pooled_vector)
        
    if len(bpms) == 0:
        print("No files were processed! Check your dataset path.")
        return
        
    X = np.vstack(embeddings) # Shape: [N, 1024]
    y = np.array(bpms)
    
    num_dims = X.shape[1]
    print(f"Dataset shape: {X.shape}. Running correlation on {num_dims} dimensions...")
    
    correlations = []
    p_values = []
    
    for i in range(num_dims):
        # Calculate Pearson correlation for dimension i
        dim_values = X[:, i]
        corr, p_val = pearsonr(dim_values, y)
        correlations.append(corr)
        p_values.append(p_val)
        
    # Save to CSV
    results_df = pd.DataFrame({
        "dimension": np.arange(num_dims),
        "correlation": correlations,
        "abs_correlation": np.abs(correlations),
        "p_value": p_values
    })
    
    results_df = results_df.sort_values(by="abs_correlation", ascending=False)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved correlation results to {OUTPUT_CSV}")
    
    print("\n--- TOP 10 CORRELATED DIMENSIONS ---")
    print(results_df.head(10))
    
    # Plot top 3 dimensions
    top_dims = results_df["dimension"].head(3).tolist()
    
    plt.figure(figsize=(10, 6))
    for dim in top_dims:
        plt.plot(y, X[:, dim], marker='o', label=f"Dim {dim} (r={correlations[dim]:.2f})")
        
    plt.title("Whisper Embeddings: Top Dimensions Correlated with BPM")
    plt.xlabel("BPM")
    plt.ylabel("Embedding Value (Mean Pooled)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Saved plot of top dimensions to {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()
