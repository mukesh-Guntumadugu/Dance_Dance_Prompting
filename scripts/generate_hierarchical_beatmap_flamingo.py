#!/usr/bin/env python3
"""
generate_hierarchical_beatmap_flamingo.py
=========================================
Phase 4 of the Hierarchical Architecture using Music-Flamingo.
"""

import os
import sys
import json
import torch
import argparse
import random
import librosa
import numpy as np
from tqdm import tqdm

from peft import PeftModel
try:
    import transformers
    from transformers import (
        AudioFlamingo3ForConditionalGeneration,
        AudioFlamingo3Processor,
    )
except ImportError:
    pass

# ── Paths ──
DICT_PATH   = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns.json"
HF_MODEL_ID = "nvidia/music-flamingo-hf"
LORA_DIR    = "/data/mg546924/models/music-flamingo-hierarchical-director"
os.environ['HF_HOME'] = "/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints"

# Hyperparameters
SAMPLE_RATE = 16000
MEASURES_PER_CHUNK = 4

def load_actor_dictionary(dict_path):
    print(f"Loading Actor Sub-Decoder dictionary from {dict_path}...")
    with open(dict_path, "r") as f:
        cluster_dict = json.load(f)
    print(f"  Loaded {len(cluster_dict)} physical cluster mappings.")
    return cluster_dict

def extract_tokens_from_response(text):
    import re
    tokens = re.findall(r"<\|cluster_\d+\|>", text)
    return tokens

def align_tokens_to_measures(tokens, target_len):
    if not tokens:
        return ["<|cluster_0|>"] * target_len
    aligned = []
    for i in range(target_len):
        idx = int(i * len(tokens) / target_len)
        aligned.append(tokens[idx])
    return aligned

def generate_beatmap(audio_path, out_ssc_path, bpm, difficulty="Challenge"):
    os.makedirs(os.path.dirname(out_ssc_path) if os.path.dirname(out_ssc_path) else ".", exist_ok=True)
    
    # 1. Load Actor Map
    cluster_dict = load_actor_dictionary(DICT_PATH)
    
    # 2. Setup Flamingo Model
    print(f"Loading Flamingo Processor from {LORA_DIR}...")
    try:
        processor = AudioFlamingo3Processor.from_pretrained(LORA_DIR, trust_remote_code=True)
    except Exception:
        processor = AudioFlamingo3Processor.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
        TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"
        with open(TOKENS_TXT, "r") as f:
            tokens = [line.strip() for line in f if line.strip()]
        processor.tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    print("Loading Base Model...")
    base_model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Resizing embeddings to {len(processor.tokenizer)}...")
    base_model.resize_token_embeddings(len(processor.tokenizer))
    
    if os.path.exists(LORA_DIR):
        print(f"Loading LoRA weights from {LORA_DIR}...")
        model = PeftModel.from_pretrained(base_model, LORA_DIR)
        model = model.eval()
    else:
        print(f"WARNING: LoRA path {LORA_DIR} not found. Running with base model.")
        model = base_model.eval()
    
    # 3. Process Audio
    print(f"\nAnalyzing Audio: {audio_path}")
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    duration = librosa.get_duration(y=y, sr=sr)
    print(f"Total Duration: {duration:.2f}s")
    
    beats_per_chunk = MEASURES_PER_CHUNK * 4
    CHUNK_SEC = beats_per_chunk * (60.0 / bpm)
    print(f"Dynamic Chunk Size: {CHUNK_SEC:.2f} seconds ({MEASURES_PER_CHUNK} measures at {bpm} BPM)")
    
    all_cluster_tokens = []
    
    # 4. Sliding Window Inference
    print("\nRunning Director Inference...")
    for win_start in np.arange(0, duration, CHUNK_SEC):
        win_end = win_start + CHUNK_SEC
        if win_end > duration:
            if (duration - win_start) < (CHUNK_SEC * 0.5):
                break
            win_end = duration
            
        start_idx = int(win_start * sr)
        end_idx   = int(win_end * sr)
        y_chunk = y[start_idx:end_idx]
        
        prompt = (
            "You are a rhythm game beatmap pattern generator. "
            f"Listen to this audio segment which corresponds exactly to {MEASURES_PER_CHUNK} measure(s) in 4/4 time. "
            f"The difficulty is {difficulty}. "
            "Predict the ordered sequence of rhythmic pattern cluster tokens."
        )
        
        text = f"User: {prompt}\nAssistant: "
        inputs = processor(text=[text], audio=[y_chunk], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9
                )
            
        input_length = inputs["input_ids"].shape[1]
        response = processor.tokenizer.decode(output_ids[0][input_length:], skip_special_tokens=True)
        tokens = extract_tokens_from_response(response)
        
        print(f"  [{win_start:04.1f}s - {win_end:04.1f}s]: {' '.join(tokens)}")
        all_cluster_tokens.extend(tokens)
        
    # 5. Math
    total_beats = duration * (bpm / 60.0)
    target_measures = int(np.round(total_beats / 4.0))
    aligned_tokens = align_tokens_to_measures(all_cluster_tokens, target_measures)
    
    # 6. Actor Translation
    physical_measures = []
    for t in aligned_tokens:
        if t in cluster_dict and cluster_dict[t]:
            pattern_str = random.choice(cluster_dict[t])
            physical_measures.append(pattern_str)
        else:
            physical_measures.append("\n".join(["0000" for _ in range(192)]))
            
    # 7. SSC Formatting
    ssc_header = f"""#VERSION:0.83;
#TITLE:Generated Flamingo Hierarchical Chart;
#MUSIC:{os.path.basename(audio_path)};
#OFFSET:0.000000;
#SAMPLESTART:0.000000;
#SAMPLELENGTH:10.000000;
#SELECTABLE:YES;
#BPMS:0.000000={bpm:.6f};
#TIMESIGNATURES:0.000000=4=4;
#TICKCOUNTS:0.000000=4;
"""
    measures_str = ",\n".join(physical_measures)
    ssc_chart = f"""
//---------------dance-single - ----------------
#NOTEDATA:;
#STEPSTYPE:dance-single;
#DIFFICULTY:{difficulty};
#METER:9;
#NOTES:
{measures_str}
;"""

    with open(out_ssc_path, "w", encoding="utf-8") as f:
        f.write(ssc_header + ssc_chart)
        
    print(f"[OK] Flamingo Hierarchical Beatmap Generated: {out_ssc_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--bpm", type=float, default=130.0)
    parser.add_argument("--difficulty", default="Challenge")
    parser.add_argument("--out", default="output.ssc")
    args = parser.parse_args()
    generate_beatmap(args.audio, args.out, args.bpm, args.difficulty)
