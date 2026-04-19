"""
verify_model_bpm.py
===================
Calculates the exact mathematical BPM of an audio track using Librosa,
and compares it against zero-shot predictions by Qwen, MuMu, DeepResonance, and Flamingo.

Usage:
    python3 onsetdetection/verify_model_bpm.py --audio_path <path.ogg> --model all
"""

import argparse
import sys
import os
import torch
import librosa
import numpy as np
import tempfile
import soundfile as sf

ROOT = "/data/mg546924/llm_beatmap_generator"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

PROMPT = "Listen to this audio clip. What is the unified global BPM (Beats Per Minute) of this song? Output only the exact integer number."

SEP = "─" * 68

def get_librosa_bpm(audio_path: str) -> float:
    print(f"  [Librosa] Calculating actual mathematical BPM from full track...")
    y, sr = librosa.load(audio_path, sr=None)
    
    # 1. Global unified tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    val = tempo[0] if isinstance(tempo, np.ndarray) else float(tempo)
    global_bpm = round(float(val), 2)
    
    # 2. Dynamic tempo changes (drift)
    dynamic_tempo, _ = librosa.beat.beat_track(y=y, sr=sr, aggregate=None)
    if isinstance(dynamic_tempo, np.ndarray) and len(dynamic_tempo) > 0:
        min_t, max_t = float(np.min(dynamic_tempo)), float(np.max(dynamic_tempo))
        print(f"  [Librosa] Changing Tempo Detected: Range [{round(min_t, 2)} to {round(max_t, 2)}] BPM")

    return global_bpm

# ── MuMu-LLaMA ────────────────────────────────────────────────────────────────
def ask_mumu_bpm(audio_path: str) -> str:
    from mumu_measure_interface import initialize_mumu_model
    import llama

    model, _ = initialize_mumu_model()
    formatted = llama.utils.format_prompt(PROMPT)

    # Use first 30s to prevent context scale overflow
    y, sr = librosa.load(audio_path, sr=24000, duration=30.0)

    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            results = model.generate(
                prompts=[formatted],
                audios=y,
                max_gen_len=16,
                temperature=0.1,
            )
    return str(results[0]).strip() if isinstance(results, list) else str(results).strip()

# ── Qwen2-Audio ───────────────────────────────────────────────────────────────
def ask_qwen_bpm(audio_path: str) -> str:
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    
    model_path = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
    )
    
    y, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate, duration=30.0)
    
    messages = [
        {"role": "system", "content": "You are a concise musical assistant."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "dummy"},
            {"type": "text", "text": PROMPT}
        ]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
    inputs = processor(text=text, audios=[y], sampling_rate=sr, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=16, temperature=0.1, do_sample=False)
        
    val = processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    
    del model
    del processor
    torch.cuda.empty_cache()
    return val.strip()

# ── DeepResonance ─────────────────────────────────────────────────────────────
def ask_deepresonance_bpm(audio_path: str) -> str:
    from deepresonance_measure_interface import initialize_deepresonance_model
    model = initialize_deepresonance_model()
    
    y, sr = librosa.load(audio_path, sr=24000, duration=30.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        tmp_path = tmp.name
        
    inputs = {
        "inputs": ["<Audio>"],
        "instructions": [PROMPT],
        "mm_names": [["audio"]],
        "mm_paths": [[os.path.basename(tmp_path)]],
        "mm_root_path": os.path.dirname(tmp_path),
        "outputs": [""],
    }
    
    try:
        resp = model.predict(inputs, max_tgt_len=16, top_p=1.0, temperature=0.1, stops_id=[[835]])
        if isinstance(resp, list): resp = resp[0] if resp else ""
        return str(resp).strip()
    finally:
        os.unlink(tmp_path)


# ── Music-Flamingo ────────────────────────────────────────────────────────────
def ask_flamingo_bpm(audio_path: str) -> str:
    from music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
    setup_music_flamingo()
    
    y, sr = librosa.load(audio_path, sr=24000, duration=30.0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, y, sr)
        tmp_path = tmp.name

    fl_prompt = f"<Audio>\n{PROMPT}"
    try:
        ans = generate_beatmap_with_flamingo(tmp_path, fl_prompt)
        return str(ans).strip()
    finally:
        os.unlink(tmp_path)

# ── Main Engine ────────────────────────────────────────────────────────────────

MODEL_FUNCS = {
    "mumu":          ("MuMu-LLaMA",     ask_mumu_bpm),
    "qwen":          ("Qwen2-Audio",    ask_qwen_bpm),
    "deepresonance": ("DeepResonance",  ask_deepresonance_bpm),
    "flamingo":      ("Music-Flamingo", ask_flamingo_bpm),
}

def main():
    parser = argparse.ArgumentParser(description="Probe models for Unified BPM estimate")
    parser.add_argument("--audio_path", required=True, help="Path to the song file")
    parser.add_argument(
        "--model",
        choices=list(MODEL_FUNCS.keys()) + ["all"],
        default="all",
        help="Which model to probe",
    )
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"File not found: {args.audio_path}")
        return

    print("\n" + SEP)
    print(f"🎵  UNIFIED BPM VERIFICATION")
    print(f"Target: {os.path.basename(args.audio_path)}")
    print(SEP + "\n")

    # 1. Math Ground Truth
    actual_bpm = get_librosa_bpm(args.audio_path)
    print(f"✅  ACTUAL (LIBROSA): {actual_bpm} BPM\n")

    targets = (
        list(MODEL_FUNCS.items())
        if args.model == "all"
        else [(args.model, MODEL_FUNCS[args.model])]
    )

    print("--- Neural Estimates ---\n")
    for key, (label, func) in targets:
        print(f"▶ {label} ...")
        try:
            bpm_ans = func(args.audio_path)
            print(f"   => {bpm_ans}")
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
        
        # Aggressive memory cleanup between heavy models if running 'all'
        if args.model == "all":
            torch.cuda.empty_cache()
            
    print("\nDone.")

if __name__ == "__main__":
    main()
