import os
import sys
import argparse
import librosa
import soundfile as sf
import torch
import numpy as np
import pickle

def extract_qwen(audio_path):
    print("Extracting Qwen...")
    import qwen_interface
    qwen_interface.setup_qwen()
    model = qwen_interface._model
    processor = qwen_interface._processor
    
    y, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
    audio_uri = f"file://{os.path.abspath(audio_path)}"
    
    # Get Prediction
    prompt = "Listen to this audio. What is the BPM?"
    bpm_text = qwen_interface.generate_beatmap_with_qwen(audio_path, prompt)
    
    # Get Latents
    conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": audio_uri}, {"type": "text", "text": "probe"}]}]
    text_context = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text_context, audio=[y], sampling_rate=sr, return_tensors="pt")
    
    with torch.no_grad():
        audio_tower = model.audio_tower
        audio_outputs = audio_tower(inputs.input_features.to(model.device), output_hidden_states=True)
        features = audio_outputs.hidden_states[-1][0].cpu().numpy()
        
    return bpm_text, features

def extract_mumu(audio_path):
    print("Extracting MuMu...")
    import mumu_measure_interface
    mumu_measure_interface.initialize_mumu_model()
    model = mumu_measure_interface._mumu_model
    
    import mumu_interface
    mumu_interface.setup_mumu()
    
    # Get Prediction
    prompt = "What is the BPM?"
    bpm_text = mumu_interface.generate_beatmap_with_mumu(audio_path, prompt)
    
    # Get Latents
    import torchaudio
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 24000:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=24000)
    audio_tensor = torch.mean(waveform, 0).unsqueeze(0).cuda()
    
    with torch.no_grad():
        features = model.encode_audio(audio_tensor)
        if isinstance(features, tuple):
            features = features[0]
        features = features.cpu().numpy()[0]
        
    return bpm_text, features

def extract_encodec(audio_path):
    print("Extracting EnCodec...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "Neural Audio Codecs"))
    from EnCodecimplementation import AudioTokenizer
    tokenizer = AudioTokenizer(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    tokens = tokenizer.tokenize(audio_path).cpu().numpy()[0]
    return "N/A (No LLM)", tokens

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen", "mumu", "encodec"])
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # We skip Flamingo and DeepResonance here for simplicity of the script unless specifically needed, 
    # but the logic scales identically.
    
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
    
    try:
        if args.model == "qwen":
            bpm, latents = extract_qwen(args.audio)
        elif args.model == "mumu":
            bpm, latents = extract_mumu(args.audio)
        elif args.model == "encodec":
            bpm, latents = extract_encodec(args.audio)
            
        with open(args.out, "wb") as f:
            pickle.dump({"bpm": bpm, "latents": latents}, f)
            
        print(f"Successfully saved {args.model} latents to {args.out}")
    except Exception as e:
        print(f"Failed to extract {args.model}: {e}")
