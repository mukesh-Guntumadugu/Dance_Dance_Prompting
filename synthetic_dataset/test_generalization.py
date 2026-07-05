#!/usr/bin/env python3
import os
import time
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from transformers import WhisperFeatureExtractor, WhisperModel
from tqdm import tqdm

def extract_embeddings(dataset_dir, model, feature_extractor, device):
    bpms = []
    embeddings = []
    
    print(f"Processing dataset: {dataset_dir}")
    for bpm in tqdm(range(60, 241)):
        # Handle different naming conventions in the two datasets
        if "linear_probing" in dataset_dir:
            wav_path = os.path.join(dataset_dir, f"linear_probing_{bpm}_bpm_stable.wav")
        else:
            wav_path = os.path.join(dataset_dir, f"bpm_{bpm}.wav")
            
        if not os.path.exists(wav_path):
            continue
            
        # Load exactly 30s
        audio, sr = librosa.load(wav_path, sr=16000, duration=30.0)
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        with torch.no_grad():
            encoder_outputs = model.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            pooled_vector = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            
        bpms.append(bpm)
        embeddings.append(pooled_vector)
        
    return np.vstack(embeddings), np.array(bpms)

def main():
    MODEL_ID = "openai/whisper-large-v2"
    STABLE_DIR = "linear_probing_dataset"
    RANDOM_DIR = "sweep_dataset"
    OUTPUT_PLOT = "Pred_vs_Target_BPM.png"
    
    # We found these dimensions in the previous experiment
    TARGET_DIMS = [220, 181, 1137]
    
    print(f"Loading {MODEL_ID}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    model = WhisperModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    
    start_time = time.time()
    
    # 1. Extract from both datasets
    print("\n--- PHASE 1: EXTRACTION ---")
    X_train_full, y_train = extract_embeddings(STABLE_DIR, model, feature_extractor, device)
    X_test_full, y_test = extract_embeddings(RANDOM_DIR, model, feature_extractor, device)
    
    if len(X_train_full) == 0 or len(X_test_full) == 0:
        print("Error: Could not find audio files in one or both datasets.")
        return
        
    # 2. Check correlations on the randomized dataset
    print("\n--- PHASE 2: INSTRUMENT GENERALIZATION ---")
    print("Testing if our top dimensions hold up when instruments change...")
    for dim in TARGET_DIMS:
        train_corr, _ = pearsonr(X_train_full[:, dim], y_train)
        test_corr, _ = pearsonr(X_test_full[:, dim], y_test)
        print(f"Dimension {dim:4d} | Stable Correlation: {train_corr:.4f} | Randomized Correlation: {test_corr:.4f}")
        
    # 3. Train a Linear Predictor using ONLY the top 3 dimensions
    print("\n--- PHASE 3: BPM PREDICTION ---")
    X_train_subset = X_train_full[:, TARGET_DIMS]
    X_test_subset = X_test_full[:, TARGET_DIMS]
    
    # Using numpy least squares to avoid requiring scikit-learn dependency
    # Add a column of ones to X for the intercept (y = mx + b)
    X_train_bias = np.c_[X_train_subset, np.ones(X_train_subset.shape[0])]
    X_test_bias = np.c_[X_test_subset, np.ones(X_test_subset.shape[0])]
    
    # Calculate weights: w = (X^T X)^-1 X^T y
    weights, residuals, rank, s = np.linalg.lstsq(X_train_bias, y_train, rcond=None)
    
    # Predict on the randomized test set!
    y_pred = X_test_bias @ weights
    
    # Calculate metrics
    mae = np.mean(np.abs(y_test - y_pred))
    print(f"\nModel trained successfully on {len(TARGET_DIMS)} dimensions!")
    print(f"Formula Weights: {weights[:-1]}")
    print(f"Formula Intercept: {weights[-1]:.2f}")
    print(f"\n=> MEAN ABSOLUTE ERROR on Randomized Dataset: {mae:.2f} BPM")
    
    # 4. Plot the results
    plt.figure(figsize=(10, 8))
    
    plt.scatter(y_test, y_pred, alpha=0.7, color='blue', label='Predictions (Randomized Instruments)')
    
    # Perfect prediction line
    plt.plot([60, 240], [60, 240], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
    
    plt.title(f"Predicting BPM with Whisper Dimensions {TARGET_DIMS}\nMAE: {mae:.2f} BPM")
    plt.xlabel("True BPM (from 60 to 240)")
    plt.ylabel("Predicted BPM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"\nSaved prediction plot to {OUTPUT_PLOT}")
    
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
