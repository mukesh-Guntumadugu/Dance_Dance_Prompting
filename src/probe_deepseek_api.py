import os
import argparse
import base64
import librosa
import soundfile as sf
from openai import OpenAI  # type: ignore
import json
import time

def encode_audio_base64(audio_path, target_sr=16000, duration=10.0):
    """
    Loads the first `duration` seconds of the audio, resamples to `target_sr`, 
    forces mono, and returns the base64 encoded string so we don't blow up the API payload.
    """
    print(f"   -> Processing {os.path.basename(audio_path)} (First {duration}s)...")
    y, sr = librosa.load(audio_path, sr=target_sr, duration=duration, mono=True)
    
    # Save to a temporary fast-buffer WAV file to encode cleanly
    temp_wav = "/tmp/deepseek_temp_probe.wav"
    sf.write(temp_wav, y, target_sr)
    
    with open(temp_wav, "rb") as f:
        audio_data = f.read()
        
    encoded = base64.b64encode(audio_data).decode("utf-8")
    
    if os.path.exists(temp_wav):
         os.remove(temp_wav)
         
    return encoded

def probe_deepseek(client, audio_b64, filename, output_dir):
    print(f"   -> Sending {filename} payload via DeepSeek API...")
    
    prompt = (
        "You are an expert Music Information Retrieval agent. Listen to the provided audio file. "
        "1. What is your estimated BPM?\n"
        "2. Return a JSON array tightly containing the exact timestamps (in seconds, e.g. 0.05, 0.42, 1.1) "
        "of all major structural drum onsets you hear in this track snippet.\n"
        "Format your answer cleanly."
    )
    
    try:
        # Utilizing the highly-standardized OpenAI compatibility wrapper for DeepSeek
        response = client.chat.completions.create(
            model="deepseek-v3.2",  # Defaulting to standard multimodal endpoint string
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "input_audio", 
                            "input_audio": {"data": audio_b64, "format": "wav"}
                        }
                    ]
                }
            ],
            temperature=0.0,
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        
    except Exception as e:
        answer = f"API Error: {str(e)}\n\n(Ensure your DEEPSEEK_API_KEY is valid and the V3.2 Multimodal backend endpoint structure accepts standard `input_audio` schema)."
        print(f"   [!] Failed: {e}")
        
    out_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_deepseek_prediction.txt")
    with open(out_file, "w") as f:
        f.write(f"Target: {filename}\n")
        f.write("-" * 50 + "\n")
        f.write(answer + "\n")
        
    print(f"✅ Saved DeepSeek Inference to: {out_file}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', type=str, required=True, help="Directory containing audio")
    parser.add_argument('--output_dir', type=str, default="results_deepseek_probe_fraxtil", help="Output directory")
    args = parser.parse_args()
    
    args.target_dir = os.path.abspath(args.target_dir)
    args.output_dir = os.path.abspath(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        print("❌ FATAL: You must set DEEPSEEK_API_KEY in your environment to use this.")
        print("Run: export DEEPSEEK_API_KEY='your-key-here'")
        return
        
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    
    supported_exts = ['.ogg', '.mp3', '.wav']
    files = []
    for root, _, filenames in os.walk(args.target_dir):
        for f in filenames:
            if os.path.splitext(f)[1].lower() in supported_exts:
                files.append(os.path.join(root, f))
                
    if not files:
        print(f"❌ No audio files found.")
        return
        
    print(f"\n========================================")
    print(f"PROBING DEEPSEEK API (V3.2 Unified Latent Space)")
    print(f"========================================")
    
    # We will only probe the first 5 songs so we don't randomly burn a massive API bill
    # You can remove `[:5]` later if everything looks perfect.
    for fp in files[:5]:
        b64_audio = encode_audio_base64(fp)
        probe_deepseek(client, b64_audio, os.path.basename(fp), args.output_dir)
        time.sleep(1.0) # rate limit protection

if __name__ == "__main__":
    main()
