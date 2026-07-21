import torch
import librosa
import numpy as np
import pandas as pd
from transformers import WhisperFeatureExtractor, WhisperModel
import os

def main():
    audio_path = "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/synthetic_dataset/sweep_dataset/bpm_120/bpm_120.wav"
    output_path = "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/outputs/whisper_encoded_1sec_120bpm.csv"
    
    print(f"Loading 1 second of audio from {audio_path}...")
    # Load exactly 1 second at 16kHz
    y, sr = librosa.load(audio_path, sr=16000, duration=1.0)
    
    print("Loading Whisper Large-V2 model (this may take a minute if downloading)...")
    model_id = "openai/whisper-large-v2"
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_id)
    model = WhisperModel.from_pretrained(model_id)
    
    print("Extracting Mel-spectrogram features...")
    inputs = feature_extractor(y, sampling_rate=sr, return_tensors="pt")
    
    print("Passing through the Whisper Encoder...")
    with torch.no_grad():
        encoder_outputs = model.encoder(inputs.input_features)
        
    last_hidden_state = encoder_outputs.last_hidden_state
    
    # Whisper pads audio to 30 seconds (1500 frames). 
    # Since we only gave 1 second, the actual audio is in the first 50 frames (50 frames/sec).
    # We will slice out just those 50 frames.
    frames_per_sec = 50
    encoded_features = last_hidden_state[0, :frames_per_sec, :].numpy()
    
    print(f"Creating CSV with shape {encoded_features.shape}...")
    df = pd.DataFrame(encoded_features)
    df.columns = [f"Dim_{i+1}" for i in range(encoded_features.shape[1])]
    df.index.name = "Frame_Index_20ms"
    
    df.to_csv(output_path)
    print(f"Successfully saved to: {output_path}")

if __name__ == "__main__":
    main()
