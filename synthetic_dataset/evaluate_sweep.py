#!/usr/bin/env python3
import os
import sys
import glob
import math
import subprocess
import tempfile
import re
import json
import csv
import librosa
import soundfile as sf
import numpy as np
import argparse

PROJ_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Point to the synthetic data folder
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_dataset")

MODELS = {
    "Librosa": {}, # Handled internally
    "Qwen": {
        "bin": "/data/mg546924/conda_envs/qwenenv/bin/python",
        "code": r'''
import os, sys, re
AUDIO = os.environ["AUDIO_PATH"]
PROMPT = os.environ["PROMPT_TEXT"]
PROJ = os.environ["PROJ_DIR"]
sys.path.insert(0, PROJ)
from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
setup_qwen()
resp_bpm = generate_beatmap_with_qwen(AUDIO, PROMPT)
nums = re.findall(r"\d+\.?\d*", str(resp_bpm))
print("BPM_RESPONSE=" + str(nums[0] if nums else "0.0"))
'''
    },
    "MuMu": {
        "bin": "/home/mg546924/.conda/envs/mumullama/bin/python",
        "code": r'''
import os, sys, re
AUDIO = os.environ["AUDIO_PATH"]
PROMPT = os.environ["PROMPT_TEXT"]
PROJ = os.environ["PROJ_DIR"]
sys.path.insert(0, PROJ)
from src.mumu_interface import setup_mumu, generate_beatmap_with_mumu
setup_mumu()
resp_bpm = generate_beatmap_with_mumu(AUDIO, PROMPT)
nums = re.findall(r"\d+\.?\d*", str(resp_bpm))
print("BPM_RESPONSE=" + str(nums[0] if nums else "0.0"))
'''
    },
    "Flamingo": {
        "bin": "/data/mg546924/music_flamingo_env/bin/python",
        "code": r'''
import os, sys, re
AUDIO = os.environ["AUDIO_PATH"]
PROMPT = os.environ["PROMPT_TEXT"]
PROJ = os.environ["PROJ_DIR"]
os.environ["HF_HOME"] = PROJ + "/Music-Flamingo/checkpoints"
sys.path.insert(0, PROJ)
from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
setup_music_flamingo()
resp_bpm = generate_beatmap_with_flamingo(AUDIO, PROMPT)
nums = re.findall(r"\d+\.?\d*", str(resp_bpm))
print("BPM_RESPONSE=" + str(nums[0] if nums else "0.0"))
'''
    },
    "DeepResonance": {
        "bin": "/data/mg546924/conda_envs/deepresonance_env/bin/python",
        "code": r'''
import os, sys, re, torch
AUDIO = os.environ["AUDIO_PATH"]
PROMPT = os.environ["PROMPT_TEXT"]
PROJ = os.environ["PROJ_DIR"]
CKPT = PROJ + "/DeepResonance/ckpt"
sys.path.insert(0, PROJ + "/DeepResonance/code")
os.chdir(PROJ + "/DeepResonance/code")
from inference_deepresonance import DeepResonancePredict

if AUDIO.endswith('.mp3'):
    import librosa
    import soundfile as sf
    import tempfile
    temp_wav = os.path.join(tempfile.gettempdir(), f"temp_dr_{os.path.basename(AUDIO)}.wav")
    y, sr = librosa.load(AUDIO, sr=16000)
    sf.write(temp_wav, y, sr)
    AUDIO = temp_wav

args = {
    "stage": 2, "mode": "test", "dataset": "musiccaps", "project_path": PROJ + "/DeepResonance/code",
    "llm_path": CKPT + "/pretrained_ckpt/vicuna_ckpt/7b_v0", "imagebind_path": CKPT + "/pretrained_ckpt/imagebind_ckpt/huge",
    "imagebind_version": "huge", "max_length": 512, "max_output_length": 512, "num_clip_tokens": 77, "gen_emb_dim": 768,
    "preencoding_dropout": 0.1, "num_preencoding_layers": 1, "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
    "freeze_lm": False, "freeze_input_proj": False, "freeze_output_proj": False, "prompt": "", "prellmfusion": True,
    "prellmfusion_dropout": 0.1, "num_prellmfusion_layers": 1, "imagebind_embs_seq": True, "topp": 1.0, "temp": 0.1,
    "ckpt_path": CKPT + "/DeepResonance_data_models/ckpt/deepresonance_beta_delta_ckpt/delta_ckpt/deepresonance/7b_tiva_v0",
}
model = DeepResonancePredict(args)
inputs = {
    "inputs": [PROMPT], "instructions": [PROMPT], "mm_names": [["audio"]],
    "mm_paths": [[os.path.basename(AUDIO)]], "mm_root_path": os.path.dirname(AUDIO), "outputs": [""],
}
resp_bpm = model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=0.1, stops_id=[[835]])
if isinstance(resp_bpm, list): resp_bpm = resp_bpm[0]
nums = re.findall(r"\d+\.?\d*", str(resp_bpm))
print("BPM_RESPONSE=" + str(nums[0] if nums else "0.0"))
'''
    }
}

def get_ground_truth_segments(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get("segments", [])

def run_model_subprocess(model_name, audio_path, prompt_text, y=None, sr=None):
    if model_name == "Librosa":
        if y is None or sr is None:
            # Load it here just in case, but prefer passing chunked y
            y, sr = librosa.load(audio_path, sr=16000)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        return float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)

    config = MODELS[model_name]
    env = os.environ.copy()
    env["AUDIO_PATH"] = audio_path
    env["PROMPT_TEXT"] = prompt_text
    env["PROJ_DIR"] = PROJ_DIR
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(config["code"])
        tmp = f.name
        
    try:
        r = subprocess.run([config["bin"], tmp], capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            print(f"    [ERROR] Subprocess crashed with return code {r.returncode}!")
            print(f"    [STDERR]: {r.stderr.strip()}")
            
        out = r.stdout + r.stderr
        m_bpm = re.search(r'BPM_RESPONSE=(.*)', out)
        if m_bpm:
            try:
                return float(m_bpm.group(1).strip())
            except: pass
        else:
            if r.returncode == 0:
                print(f"    [WARNING] BPM_RESPONSE not found in output!")
                print(f"    [STDOUT]: {r.stdout.strip()}")
        return None
    except Exception as e:
        print(f"Error querying {model_name}: {e}")
        return None
    finally:
        os.remove(tmp)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODELS.keys()), help="Which model to run")
    parser.add_argument("--mode", type=str, required=True, choices=["stateless_chunk", "true_history", "fake_history", "full_song"], 
                        help="C0: stateless_chunk, C1: true_history, C2: fake_history, C3: full_song")
    # For synthetic dataset, we allow specifying extension, defaulting to wav
    parser.add_argument("--ext", type=str, default="wav", choices=["wav", "ogg", "mp3"])
    parser.add_argument("--run_idx", type=str, default="", help="Optional run index to append to output files")
    args = parser.parse_args()
    
    run_suffix = f"_run{args.run_idx}" if args.run_idx else ""
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{args.model}_{args.mode}_{args.ext}{run_suffix}_rmse.csv")
    out_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{args.model}_{args.mode}_{args.ext}{run_suffix}_report.json")
    
    songs = []
    if not os.path.exists(DATASET_DIR):
        print(f"Dataset dir does not exist: {DATASET_DIR}")
        print("Please run build_sweep_datasetset.sh first!")
        return
        
    for d in os.listdir(DATASET_DIR):
        full_dir = os.path.join(DATASET_DIR, d)
        if os.path.isdir(full_dir):
            audio_path = os.path.join(full_dir, f"{d}.{args.ext}")
            json_path = os.path.join(full_dir, f"{d}_groundtruth.json")
            if os.path.exists(audio_path) and os.path.exists(json_path):
                try:
                    # e.g., bpm_60 -> 60
                    num = int(d.split('_')[1])
                except:
                    num = 999
                songs.append((num, d, audio_path, json_path))
                
    songs = sorted(songs, key=lambda x: x[0])
    print(f"Running {args.model} in {args.mode} mode over {len(songs)} synthetic songs (format: {args.ext})...")
    
    results = []
    
    processed_songs = set()
    mode_open = "w"
    if os.path.exists(out_csv):
        mode_open = "a"
        with open(out_csv, "r") as f:
            for line in f.readlines()[1:]:
                processed_songs.add(line.split(",")[0])
    
    with open(out_csv, mode_open, newline="") as csv_f:
        writer = csv.writer(csv_f)
        if mode_open == "w":
            writer.writerow(["song_name", "window_start", "window_end", "actual_bpm", "pred_bpm"])
        
        for num, song_name, audio_path, json_path in songs:
            if song_name in processed_songs:
                print(f"Skipping {song_name}, already processed.")
                continue
                
            print(f"\nProcessing {song_name}...")
            
            y, sr = None, 16000
            
            with open(json_path, 'r') as json_f:
                gt_data = json.load(json_f)
            duration = gt_data.get("duration", 180.0)
            segments = gt_data.get("segments", [])
            
            if args.mode != "full_song" or args.model == "Librosa":
                try:
                    y, sr = librosa.load(audio_path, sr=16000)
                except Exception as e:
                    print(f"Failed to load audio {audio_path}: {e}")
                    continue
            
            # Dynamic chunking logic: split at tempo changes exactly, max 40s per chunk
            chunk_boundaries = []
            current_time = 0.0
            
            for seg in segments:
                seg_start = max(current_time, seg["start"])
                seg_end = min(duration, seg["end"])
                if seg_start >= duration:
                    break
                
                # Chop into 20s pieces
                while seg_start < seg_end:
                    chunk_end = min(seg_start + 20.0, seg_end)
                    if chunk_end - seg_start >= 1.0: # Skip micro-chunks
                        chunk_boundaries.append((seg_start, chunk_end, seg["bpm"]))
                    seg_start = chunk_end
            
            song_data = {"song_name": song_name, "windows": []}
            
            prev_pred_bpm = None
            prev_actual_bpm = None
            
            for (win_start, win_end, actual_bpm) in chunk_boundaries:
                print(f"  Window {win_start:.1f}s to {win_end:.1f}s | Target: {actual_bpm:.1f}")
                
                if args.mode in ["stateless_chunk", "true_history", "fake_history"]:
                    assert y is not None
                    y_chunk = y[int(win_start * sr):int(win_end * sr)]
                    with tempfile.NamedTemporaryFile(suffix=f".{args.ext}", delete=False) as tmp_audio:
                        sf.write(tmp_audio.name, y_chunk, sr)
                        chunk_path = tmp_audio.name
                    
                    if args.mode == "stateless_chunk":
                        prompt = "You are estimating BPM from an audio segment. Analyze only the provided audio chunk. Do not assume anything about previous or future chunks. Return only one BPM number."
                    elif args.mode == "true_history":
                        if prev_pred_bpm is not None:
                            prompt = f"Previous chunk BPM estimate: {prev_pred_bpm:.1f} BPM. Now analyze this next audio chunk. Should the BPM remain the same or change? Return the BPM for this chunk. Only output the number."
                        else:
                            prompt = "You are estimating BPM from an audio segment. Analyze only the provided audio chunk. Return only one BPM number."
                    elif args.mode == "fake_history":
                        if prev_actual_bpm is not None:
                            fake_bpm = 90.0 if prev_actual_bpm >= 120 else 140.0
                            prompt = f"Previous chunk BPM estimate: {fake_bpm:.1f} BPM. Now analyze this audio chunk independently and return the BPM. Only output the number."
                        else:
                            prompt = "You are estimating BPM from an audio segment. Analyze only the provided audio chunk. Return only one BPM number."
                    
                    pred_bpm = run_model_subprocess(args.model, chunk_path, prompt, y=y_chunk, sr=sr)
                    os.remove(chunk_path)
                else: # full_song
                    prompt = f"Here is the full song. Estimate BPM for the window between {win_start:.1f} seconds and {win_end:.1f} seconds. Only output the number."
                    pred_bpm = run_model_subprocess(args.model, audio_path, prompt, y=y, sr=sr)
                
                pred_bpm_val = pred_bpm if pred_bpm is not None else 0.0
                print(f"    -> Predicted: {pred_bpm_val:.1f}")
                
                prev_pred_bpm = pred_bpm_val
                prev_actual_bpm = actual_bpm
                
                row = [song_name, win_start, win_end, actual_bpm, pred_bpm_val]
                writer.writerow(row)
                csv_f.flush()
                
                song_data["windows"].append({
                    "start": win_start, "end": win_end, 
                    "actual_bpm": actual_bpm, "pred_bpm": pred_bpm_val
                })
                
            results.append(song_data)
            
    # Compute RMSE
    print("\n--- RMSE REPORT ---")
    all_sq_errs = []
    report = {}
    
    for song_data in results:
        song_name = song_data["song_name"]
        song_sq_errs = []
        
        for w in song_data["windows"]:
            actual = w["actual_bpm"]
            pred = w["pred_bpm"]
            if pred > 0:
                sq_err = (pred - actual) ** 2
                song_sq_errs.append(sq_err)
                all_sq_errs.append(sq_err)
        
        if song_sq_errs:
            report[song_name] = math.sqrt(sum(song_sq_errs) / len(song_sq_errs))
        else:
            report[song_name] = None
            
    if all_sq_errs:
        overall_rmse = math.sqrt(sum(all_sq_errs) / len(all_sq_errs))
        report["OVERALL"] = overall_rmse
        print(f"Overall RMSE: {overall_rmse:.2f}")
    
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)

if __name__ == "__main__":
    main()
