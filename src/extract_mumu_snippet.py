import librosa
import torchaudio
import torch
import warnings
warnings.filterwarnings('ignore')

import mumu_measure_interface

file_path = "src/test_slice.ogg"
waveform, sample_rate = torchaudio.load(file_path)
if sample_rate != 24000:
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=24000)

audio_tensor = torch.mean(waveform, 0).unsqueeze(0)
device = mumu_measure_interface._mumu_model.device
audio_tensor = audio_tensor.to(device)

model = mumu_measure_interface._mumu_model

print("Extracting MERT features...")
with torch.no_grad():
    features = model.encode_audio(audio_tensor)
    if isinstance(features, tuple):
        features = features[0]
    features = features.cpu().numpy()

# Shape is usually [1, time_steps, hidden_dim]
features = features[0] # [time_steps, hidden_dim]

import csv
with open("src/Neural Audio Codecs/outputs/mumu_features_snippet.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["Frame"] + [f"Dim_{i+1}" for i in range(features.shape[1])]
    writer.writerow(header)
    for i in range(5): # Just save 5 frames
        writer.writerow([i] + features[i].tolist())

print(f"MERT features shape: {features.shape}")
