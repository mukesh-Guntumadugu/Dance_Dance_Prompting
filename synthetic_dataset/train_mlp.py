#!/usr/bin/env python3
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import WhisperFeatureExtractor, WhisperModel
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# --- Define the Neural Network ---
class BPMNet(nn.Module):
    def __init__(self, input_dim=1024):
        super(BPMNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1) # Output is a single continuous value (BPM)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

def extract_embeddings_for_dataset(dataset_dir, model, feature_extractor, device):
    bpms = []
    embeddings = []
    
    print(f"Extracting embeddings from: {dataset_dir}")
    # The sweep datasets go from 60 to 240 BPM
    for bpm in tqdm(range(60, 241)):
        # Handle the two different directory structures
        if "linear_probing" in dataset_dir:
            wav_path = os.path.join(dataset_dir, f"linear_probing_{bpm}_bpm_stable.wav")
        else:
            wav_path = os.path.join(dataset_dir, f"bpm_{bpm}", f"bpm_{bpm}.wav")
            
        if not os.path.exists(wav_path):
            continue
            
        # Load exactly 30s of audio to be consistent
        audio, sr = librosa.load(wav_path, sr=16000, duration=30.0)
        inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(device)
        
        with torch.no_grad():
            encoder_outputs = model.encoder(input_features)
            hidden_states = encoder_outputs.last_hidden_state
            pooled_vector = hidden_states.mean(dim=1).squeeze(0).cpu().numpy()
            
        bpms.append(bpm)
        embeddings.append(pooled_vector)
        
    if len(embeddings) == 0:
        return np.array([]), np.array([])
        
    return np.vstack(embeddings), np.array(bpms)

def main():
    MODEL_ID = "openai/whisper-large-v2"
    STABLE_DIR = "linear_probing_dataset"
    RANDOM_DIR = "sweep_dataset"
    OUTPUT_PLOT = "MLP_Pred_vs_Target_BPM.png"
    MODEL_SAVE_PATH = "bpm_mlp.pth"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading {MODEL_ID} for extraction...")
    feature_extractor = WhisperFeatureExtractor.from_pretrained(MODEL_ID)
    whisper_model = WhisperModel.from_pretrained(MODEL_ID).to(device)
    whisper_model.eval()
    
    start_time = time.time()
    
    # --- PHASE 1: DATASET PREPARATION ---
    print("\n--- PHASE 1: EXTRACTION ---")
    X_stable, y_stable = extract_embeddings_for_dataset(STABLE_DIR, whisper_model, feature_extractor, device)
    X_random, y_random = extract_embeddings_for_dataset(RANDOM_DIR, whisper_model, feature_extractor, device)
    
    if len(X_stable) == 0 or len(X_random) == 0:
        print("Error: Could not load datasets.")
        return
        
    # Combine datasets
    X_all = np.vstack([X_stable, X_random])
    y_all = np.concatenate([y_stable, y_random])
    print(f"\nTotal dataset size: {len(y_all)} songs")
    
    # Split into 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.FloatTensor(y_train).to(device)
    X_test_t = torch.FloatTensor(X_test).to(device)
    y_test_t = torch.FloatTensor(y_test).to(device)
    
    # --- PHASE 2: TRAINING THE MLP ---
    print("\n--- PHASE 2: TRAINING MLP ---")
    mlp = BPMNet(input_dim=1024).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(mlp.parameters(), lr=0.001)
    
    epochs = 1000
    best_test_mae = float('inf')
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        mlp.train()
        optimizer.zero_grad()
        
        # Forward pass
        predictions = mlp(X_train_t)
        loss = criterion(predictions, y_train_t)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Evaluate on test set
        if (epoch + 1) % 100 == 0:
            mlp.eval()
            with torch.no_grad():
                test_preds = mlp(X_test_t)
                test_loss = criterion(test_preds, y_test_t)
                test_mae = torch.mean(torch.abs(test_preds - y_test_t)).item()
                
                print(f"Epoch {epoch+1:4d}/{epochs} | Train MSE: {loss.item():.2f} | Test MSE: {test_loss.item():.2f} | Test MAE: {test_mae:.2f} BPM")
                
                # Save best model
                if test_mae < best_test_mae:
                    best_test_mae = test_mae
                    torch.save(mlp.state_dict(), MODEL_SAVE_PATH)
                    
    print(f"\nTraining Complete! Best Test MAE: {best_test_mae:.2f} BPM")
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # --- PHASE 3: EVALUATION & PLOTTING ---
    print("\n--- PHASE 3: FINAL EVALUATION ---")
    # Load best model for plotting
    mlp.load_state_dict(torch.load(MODEL_SAVE_PATH))
    mlp.eval()
    
    with torch.no_grad():
        final_preds = mlp(X_test_t).cpu().numpy()
        true_vals = y_test_t.cpu().numpy()
        
    final_mae = np.mean(np.abs(final_preds - true_vals))
    print(f"Final Neural Network MAE on unseen test data: {final_mae:.2f} BPM")
    
    # Plotting
    plt.figure(figsize=(10, 8))
    plt.scatter(true_vals, final_preds, alpha=0.7, color='purple', label=f'MLP Predictions (MAE: {final_mae:.2f})')
    plt.plot([60, 240], [60, 240], color='red', linestyle='--', linewidth=2, label='Perfect Prediction')
    
    plt.title(f"PyTorch MLP BPM Predictor\nTested on Mixed Synthetic Instruments")
    plt.xlabel("True BPM")
    plt.ylabel("Predicted BPM")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT)
    print(f"Saved prediction plot to {OUTPUT_PLOT}")
    
    print(f"Total time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
