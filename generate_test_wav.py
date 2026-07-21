import librosa
import soundfile as sf
import numpy as np

def generate_click_track(bpm=120, duration=10, sr=44100):
    print(f"Generating a perfect {bpm} BPM click track...")
    
    # Calculate the beat times.
    
    # But wait, librosa.clicks takes times. Let's calculate the beat times.
    beat_interval = 60.0 / bpm
    times = np.arange(0, duration, beat_interval)
    
    y = librosa.clicks(times=times, sr=sr, length=int(sr * duration), click_freq=1000.0, click_duration=0.1)
    
    # Add some slight noise so it's not totally empty (helps some encoders)
    noise = np.random.normal(0, 0.005, len(y))
    y = y + noise
    
    # Save it
    filename = f"test_{bpm}bpm.wav"
    sf.write(filename, y, sr)
    print(f"Saved to {filename}!")

if __name__ == "__main__":
    generate_click_track(120, 10)
