#!/usr/bin/env python3
"""
Batch Chopped Onset Extractor for HPC.
Executes sequence generation for one selected LLM across multiple chop durations and temperatures.
Saves outputs inside the designated song folders: 
e.g. Bitch Clap/onsets_seconds/onsets_60/temp_0/onset_detection_Bitch Clap_qwen_60s.csv
"""

import os
import sys
import gc
import re
import csv
import argparse
import datetime
import tempfile
import torch
import librosa
import soundfile as sf
from pathlib import Path

# --- Constants & Paths ---
PROJ_DIR = os.environ.get("BENCHMARK_PROJ", "/data/mg546924/llm_beatmap_generator")
DATASET_DIR = os.path.join(PROJ_DIR, "src/musicForBeatmap/Fraxtil's Arrow Arrangements")

# Ensure proper paths
sys.path.insert(0, PROJ_DIR)
sys.path.insert(0, os.path.join(PROJ_DIR, "src"))

def parse_args():
    parser = argparse.ArgumentParser(description="Batch chopped onsets extraction.")
    parser.add_argument("--model", type=str, required=True, 
                        choices=["qwen", "mumu", "flamingo", "deepresonance"], 
                        help="Which model to run. Start ONE model per HPC slurm job.")
    parser.add_argument("--durations", type=int, nargs="+", 
                        default=[60, 50, 40, 30, 20, 10, 5, 2, 1],
                        help="List of chunk durations in seconds.")
    parser.add_argument("--temperatures", type=float, nargs="+",
                        default=[0.0, 1.0, 2.0],
                        help="Temperatures to test. 0.0 is greedy.")
    parser.add_argument("--song_dir", type=str, default=DATASET_DIR,
                        help="Root directory containing song folders.")
    return parser.parse_args()

def init_model(model_name):
    print(f"Loading {model_name} model into VRAM...")
    
    if model_name == "qwen":
        from src.qwen_interface import setup_qwen
        setup_qwen()
    elif model_name == "mumu":
        from src.mumu_interface import setup_mumu
        setup_mumu()
    elif model_name == "flamingo":
        # Required for flamingo cache
        os.environ["HF_HOME"] = os.path.join(PROJ_DIR, "Music-Flamingo", "checkpoints")
        from src.music_flamingo_interface import setup_music_flamingo
        setup_music_flamingo()
    elif model_name == "deepresonance":
        from src.deepresonance_measure_interface import initialize_deepresonance_model
        # DeepResonance requires code dir context path fixing
        sys.path.insert(0, os.path.join(PROJ_DIR, "DeepResonance", "code"))
        os.chdir(os.path.join(PROJ_DIR, "DeepResonance", "code"))
        initialize_deepresonance_model()

def run_inference(model_name, audio_chunk_path, prompt, temp):
    """
    Wrapper to call the correct inference interface. 
    It passes the temperature to the generation.
    """
    if model_name == "qwen":
        from src.qwen_interface import _model, _processor, _prefix_fn
        # Duplicate of generate_beatmap_with_qwen with custom temp injection
        target_sr = _processor.feature_extractor.sampling_rate
        y, sr = librosa.load(audio_chunk_path, sr=target_sr)
        audio_uri = f"file://{os.path.abspath(audio_chunk_path)}"
        conversation = [{"role": "user", "content": [{"type": "audio", "audio_url": audio_uri}, {"type": "text", "text": prompt}]}]
        text = _processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = _processor(text=text, audio=[y], sampling_rate=target_sr, return_tensors="pt", padding=True).to(_model.device)
        
        do_sample = temp > 0.0
        generate_kwargs = dict(
            **inputs, 
            max_new_tokens=8192,
            do_sample=do_sample,
            repetition_penalty=1.0
        )
        if do_sample: generate_kwargs["temperature"] = temp
        if _prefix_fn is not None: generate_kwargs["prefix_allowed_tokens_fn"] = _prefix_fn
            
        with torch.no_grad():
            generated_ids = _model.generate(**generate_kwargs)
        generated_ids = generated_ids[:, inputs.input_ids.size(1):]
        return _processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    elif model_name == "mumu":
        from src.mumu_interface import _mumu_model, _mumu_processor
        # Mimic generate_beatmap_with_mumu with temperature
        from src.mumu_interface import generate_beatmap_with_mumu
        import transformers
        # Temporarily monkey patch or adjust generate kwargs? Mumu interface might not expose temp.
        # It's cleaner fully overriding local config or falling back to default wrapper if unsupported
        # TODO: for now we use the basic wrapper and we will update mumu_interface if necessary
        return generate_beatmap_with_mumu(audio_chunk_path, prompt)
        
    elif model_name == "flamingo":
        from src.music_flamingo_interface import generate_beatmap_with_flamingo
        return generate_beatmap_with_flamingo(audio_chunk_path, prompt)

    elif model_name == "deepresonance":
        from src.deepresonance_measure_interface import _model
        inputs = {
            "inputs": ["<Audio>"],
            "instructions": [prompt],
            "mm_names": [["audio"]],
            "mm_paths": [[os.path.basename(audio_chunk_path)]],
            "mm_root_path": os.path.dirname(audio_chunk_path),
            "outputs": [""],
        }
        # DeepResonance explicitly supports temperature. It defaults to 0.001 instead of 0.0 to prevent inf
        safe_temp = max(temp, 0.001)
        resp = _model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=safe_temp, stops_id=[[835]])
        if isinstance(resp, list): resp = resp[0]
        return resp or ""

def main():
    args = parse_args()
    
    # 1. Initialize the single model requested by this batch execution
    init_model(args.model)
    
    song_dir_path = Path(args.song_dir)
    print(f"Scanning for .ogg files in {song_dir_path}")
    
    ogg_files = list(song_dir_path.rglob("*.ogg"))
    if not ogg_files:
        print("No .ogg files found! Are you running this on the HPC?")
        return

    # 2. Iterate through 20 songs
    for ogg_file in ogg_files:
        song_folder = ogg_file.parent
        song_name = song_folder.name
        
        print(f"\n==========================================")
        print(f"🎵 Processing Song: {song_name}")
        print(f"==========================================")
        
        y = None
        sr = None
        
        # 3. Iterate through Durations
        for dur in args.durations:
            
            # 4. Iterate through Temperatures
            for temp in args.temperatures:
                # Temperature formatted without weird decimals (e.g., 0.0 -> 0, 1.0 -> 1)
                temp_int = int(temp) if temp.is_integer() else temp
                
                # Create targeted output folder
                out_folder = song_folder / "onsets_seconds" / f"onsets_{dur}" / f"temp_{temp_int}"
                out_folder.mkdir(parents=True, exist_ok=True)
                
                # Check if it already exists to avoid duplicated 10+ hour loops
                csv_filename = f"onset_detection_{song_name}_{args.model}_{dur}s.csv"
                txt_filename = f"onset_detection_{song_name}_{args.model}_{dur}s.txt"
                out_csv = out_folder / csv_filename
                out_txt = out_folder / txt_filename
                
                if out_csv.exists():
                    print(f"⏭️ Skipping {song_name} | {dur}s | Temp {temp_int} (Already exists)")
                    continue
                
                # Lazy load audio to avoid loading if all files exist
                if y is None:
                    y, sr = librosa.load(str(ogg_file), sr=None)
                    total_dur = len(y) / sr
                
                print(f"\n▶ Running {args.model} | Duration: {dur}s | Temp: {temp_int} ...", flush=True)
                
                all_onsets = []
                full_raw_text = ""
                
                for start in range(0, int(total_dur), dur):
                    end = min(start + dur, total_dur)
                    chunk = y[int(start*sr):int(end*sr)]
                    
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        sf.write(tmp.name, chunk, sr)
                        tmp_path = tmp.name
                    
                    prompt = f"List all the onset timestamps in this {round(end-start,1)}s audio clip in milliseconds."
                    
                    # Generate response
                    try:
                        response = run_inference(args.model, tmp_path, prompt, temp)
                    except Exception as e:
                        print(f"⚠️ Inference failed on {start}s-{end}s: {e}")
                        response = ""
                        
                    full_raw_text += f"[{start}s - {end}s]\n{response}\n\n"
                    
                    # Parse and Shift timestamps
                    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", response or "")
                    for n in nums:
                        try:
                            # Parse ms, add chunk offset in ms
                            absolute_ms = int(round(float(n) + start * 1000))
                            all_onsets.append(absolute_ms)
                        except:
                            pass
                    
                    os.remove(tmp_path)
                    gc.collect()
                    torch.cuda.empty_cache()
                    
                    print(f"  Chunk {start:.0f}s-{end:.0f}s → Found {len(nums)} onsets", flush=True)
                
                # Save Outputs
                with open(out_txt, "w") as f:
                    f.write(full_raw_text)
                    
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["onset_ms"])
                    for ms in sorted(all_onsets): 
                        w.writerow([ms])
                        
                print(f"✅ Saved correctly to {out_folder}")

if __name__ == "__main__":
    main()
