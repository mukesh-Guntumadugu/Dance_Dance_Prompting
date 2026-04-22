import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio_path = "src/musicForBeatmap/Fraxtil's Arrow Arrangements/Love to Rise in the Summer Morning/Love to Rise in the Summer Morning.ogg"
print(f"Loading audio from {audio_path}...")
y, sr = librosa.load(audio_path, duration=10.0, offset=60.0)

print("Detecting onsets...")
onset_env = librosa.onset.onset_strength(y=y, sr=sr)
onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

plt.figure(figsize=(14, 5))
# For librosa standard waveshow
librosa.display.waveshow(y, sr=sr, alpha=0.6, color='black')
plt.vlines(onset_times, -1, 1, color='r', linestyle='--', alpha=0.9, linewidth=2, label='Detected Onsets')
plt.title('Audio Waveform with Detected Onsets', fontsize=16, fontweight='bold')
plt.xlabel('Time (seconds relative to offset window)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')
plt.tight_layout()

output_path = "/Users/mukeshguntumadugu/LLM_rock/llm_beatmap_generator/single_song_onsets.png"
plt.savefig(output_path, dpi=300)
print(f"Saved chart to {output_path}")
