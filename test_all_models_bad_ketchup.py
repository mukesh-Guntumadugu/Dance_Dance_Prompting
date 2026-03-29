#!/usr/bin/env python3
"""
test_all_models_bad_ketchup.py
==============================
Runs ALL open-source onset detection models on just ONE song: Bad Ketchup.
This is a quick benchmark to compare how each model performs on the same audio.

Models tested:
  1. Librosa        (traditional signal processing baseline)
  2. DeepResonance  (LLaMA 7B + ImageBind, deepresonance_env)
  3. Qwen Audio     (Qwen2-Audio LoRA, qwenenv)
  4. MuMu-LLaMA    (MuMu multimodal LLaMA)
  5. Music-Flamingo (NVIDIA 30GB multimodal, music_flamingo_env)

Output: Results printed to stdout + saved per model in the Bad Ketchup folder.
Run via: slurm_test_all_models.sh
"""

import os
import sys
import csv
import time
import datetime
import subprocess

AUDIO_PATH = "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup/Bad Ketchup.ogg"
OUT_DIR    = "/data/mg546924/llm_beatmap_generator/src/musicForBeatmap/Fraxtil's Arrow Arrangements/Bad Ketchup"
PROJ_DIR   = "/data/mg546924/llm_beatmap_generator"

ENVS = {
    "deepresonance": "/data/mg546924/conda_envs/deepresonance_env/bin/python",
    "qwen":          "/data/mg546924/conda_envs/qwenenv/bin/python",
    "flamingo":      "/data/mg546924/music_flamingo_env/bin/python",
}

results = {}

# ── 1. LIBROSA (runs in current env) ─────────────────────────────────────────
print("\n" + "="*60)
print("1/5  LIBROSA  (signal processing baseline)")
print("="*60)
try:
    import librosa
    import numpy as np

    y, sr = librosa.load(AUDIO_PATH, sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, units='time')
    onsets_ms = [round(float(t) * 1000, 1) for t in onset_frames]

    ts = datetime.datetime.now().strftime("%d%m%Y%H%M%S")
    out_file = os.path.join(OUT_DIR, f"Librosa_onsets_Bad_Ketchup_{ts}.csv")
    with open(out_file, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["onset_ms"])
        for ms in onsets_ms:
            w.writerow([ms])

    results["Librosa"] = len(onsets_ms)
    print(f"✅ Librosa found {len(onsets_ms)} onsets → {out_file}")
except Exception as e:
    results["Librosa"] = f"ERROR: {e}"
    print(f"❌ Librosa failed: {e}")


# ── 2. DEEPRESONANCE ──────────────────────────────────────────────────────────
print("\n" + "="*60)
print("2/5  DEEPRESONANCE  (LLaMA 7B + ImageBind)")
print("="*60)
dr_script = f"""
import os, sys, gc, csv, datetime, soundfile as sf, numpy as np, tempfile, torch
sys.path.insert(0, '{PROJ_DIR}/DeepResonance/code')
os.chdir('{PROJ_DIR}/DeepResonance/code')

from inference_deepresonance import DeepResonancePredict

CKPT = '{PROJ_DIR}/DeepResonance/ckpt'
args = {{
    'stage': 2, 'mode': 'test', 'dataset': 'musiccaps',
    'project_path': '{PROJ_DIR}/DeepResonance/code',
    'llm_path': CKPT + '/pretrained_ckpt/vicuna_ckpt/7b_v0',
    'imagebind_path': CKPT + '/pretrained_ckpt/imagebind_ckpt/huge',
    'imagebind_version': 'huge',
    'max_length': 512, 'max_output_length': 512,
    'num_clip_tokens': 77, 'gen_emb_dim': 768,
    'preencoding_dropout': 0.1, 'num_preencoding_layers': 1,
    'lora_r': 32, 'lora_alpha': 32, 'lora_dropout': 0.1,
    'freeze_lm': False, 'freeze_input_proj': False, 'freeze_output_proj': False,
    'prompt': '', 'prellmfusion': True, 'prellmfusion_dropout': 0.1,
    'num_prellmfusion_layers': 1, 'imagebind_embs_seq': True, 'topp': 1.0, 'temp': 0.1,
    'ckpt_path': CKPT + '/DeepResonance_data_models/ckpt/deepresonance_beta_delta_ckpt/delta_ckpt/deepresonance/7b_tiva_v0',
}}

import librosa
y, sr = librosa.load('{AUDIO_PATH}', sr=None)
duration = len(y) / sr
CHUNK = 5
chunks = [(i, min(i+CHUNK, duration)) for i in range(0, int(duration), CHUNK)]

model = DeepResonancePredict(args)

import tempfile, re, json
all_onsets = []
for (start, end) in chunks:
    audio_chunk = y[int(start*sr):int(end*sr)]
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
        import soundfile as sf
        sf.write(tmp.name, audio_chunk, sr)
        tmp_path = tmp.name
    chunk_dur = end - start
    prompt = "Describe the auditory characteristics of this music in a few words."
    inputs = {{
        "inputs": ["<Audio>"],
        "instructions": [f"List all onset timestamps in milliseconds for this {{round(chunk_dur,1)}}s clip."],
        "mm_names": [["audio"]],
        "mm_paths": [[os.path.basename(tmp_path)]],
        "mm_root_path": os.path.dirname(tmp_path),
        "outputs": [""],
    }}
    resp = model.predict(inputs, max_tgt_len=512, top_p=1.0, temperature=0.1, stops_id=[[835]])
    if isinstance(resp, list): resp = resp[0]
    nums = re.findall(r'\\b(\\d+(?:\\.\\d+)?)\\b', resp or '')
    for n in nums:
        all_onsets.append(round(float(n) + start*1000, 1))
    os.remove(tmp_path)
    gc.collect(); torch.cuda.empty_cache()

ts = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
out = '{OUT_DIR}/DeepResonance_TEST_Bad_Ketchup_' + ts + '.csv'
with open(out, 'w', newline='') as f:
    w = __import__('csv').writer(f)
    w.writerow(['onset_ms'])
    for ms in all_onsets:
        w.writerow([ms])
print(f'ONSET_COUNT={{len(all_onsets)}}')
print(f'OUTPUT_FILE={{out}}')
"""

try:
    r = subprocess.run(
        [ENVS["deepresonance"], "-c", dr_script],
        capture_output=True, text=True, timeout=1200
    )
    out = r.stdout + r.stderr
    import re
    m = re.search(r'ONSET_COUNT=(\d+)', out)
    count = int(m.group(1)) if m else "?"
    results["DeepResonance"] = count
    print(f"✅ DeepResonance found {count} onsets")
    if r.returncode != 0:
        print(f"STDERR: {r.stderr[-500:]}")
except Exception as e:
    results["DeepResonance"] = f"ERROR: {e}"
    print(f"❌ DeepResonance failed: {e}")


# ── 3. QWEN ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("3/5  QWEN AUDIO  (Qwen2-Audio LoRA)")
print("="*60)
qwen_script = f"""
import sys, os
sys.path.insert(0, '{PROJ_DIR}')
sys.path.insert(0, '{PROJ_DIR}/src')
os.chdir('{PROJ_DIR}')

from src.qwen_interface import setup_qwen, generate_beatmap_with_qwen
import csv, datetime, re

setup_qwen()
prompt = "List all the onset timestamps in this audio in milliseconds."
response = generate_beatmap_with_qwen('{AUDIO_PATH}', prompt)
nums = re.findall(r'\\b(\\d+(?:\\.\\d+)?)\\b', response or '')
onsets = [float(n) for n in nums]

ts = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
out = '{OUT_DIR}/Qwen_TEST_Bad_Ketchup_' + ts + '.csv'
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['onset_ms'])
    for ms in onsets: w.writerow([ms])
print(f'ONSET_COUNT={{len(onsets)}}')
print(f'OUTPUT_FILE={{out}}')
"""

try:
    r = subprocess.run(
        [ENVS["qwen"], "-c", qwen_script],
        capture_output=True, text=True, timeout=600
    )
    out = r.stdout + r.stderr
    import re
    m = re.search(r'ONSET_COUNT=(\d+)', out)
    count = int(m.group(1)) if m else "?"
    results["Qwen"] = count
    print(f"✅ Qwen found {count} onsets")
    if r.returncode != 0:
        print(f"STDERR: {r.stderr[-500:]}")
except Exception as e:
    results["Qwen"] = f"ERROR: {e}"
    print(f"❌ Qwen failed: {e}")


# ── 4. MuMu-LLaMA ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("4/5  MuMu-LLaMA")
print("="*60)
mumu_script = f"""
import sys, os
sys.path.insert(0, '{PROJ_DIR}')
sys.path.insert(0, '{PROJ_DIR}/src')
os.chdir('{PROJ_DIR}')

from src.mumu_interface import setup_mumu, generate_beatmap_with_mumu
import csv, datetime, re

setup_mumu()
prompt = "List all the onset timestamps in this audio in milliseconds."
response = generate_beatmap_with_mumu('{AUDIO_PATH}', prompt)
nums = re.findall(r'\\b(\\d+(?:\\.\\d+)?)\\b', response or '')
onsets = [float(n) for n in nums]

ts = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
out = '{OUT_DIR}/MuMu_TEST_Bad_Ketchup_' + ts + '.csv'
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['onset_ms'])
    for ms in onsets: w.writerow([ms])
print(f'ONSET_COUNT={{len(onsets)}}')
print(f'OUTPUT_FILE={{out}}')
"""

try:
    r = subprocess.run(
        [ENVS["deepresonance"], "-c", mumu_script],
        capture_output=True, text=True, timeout=600
    )
    out = r.stdout + r.stderr
    import re
    m = re.search(r'ONSET_COUNT=(\d+)', out)
    count = int(m.group(1)) if m else "?"
    results["MuMu-LLaMA"] = count
    print(f"✅ MuMu-LLaMA found {count} onsets")
    if r.returncode != 0:
        print(f"STDERR: {r.stderr[-500:]}")
except Exception as e:
    results["MuMu-LLaMA"] = f"ERROR: {e}"
    print(f"❌ MuMu-LLaMA failed: {e}")


# ── 5. MUSIC-FLAMINGO ─────────────────────────────────────────────────────────
print("\n" + "="*60)
print("5/5  MUSIC-FLAMINGO  (NVIDIA 30GB)")
print("="*60)
flamingo_script = f"""
import sys, os
sys.path.insert(0, '{PROJ_DIR}')
sys.path.insert(0, '{PROJ_DIR}/src')
os.environ['HF_HOME'] = '{PROJ_DIR}/Music-Flamingo/checkpoints'
os.chdir('{PROJ_DIR}')

from src.music_flamingo_interface import setup_music_flamingo, generate_beatmap_with_flamingo
import csv, datetime, re

setup_music_flamingo()
prompt = "List all the onset timestamps in this audio in milliseconds."
response = generate_beatmap_with_flamingo('{AUDIO_PATH}', prompt)
nums = re.findall(r'\\b(\\d+(?:\\.\\d+)?)\\b', response or '')
onsets = [float(n) for n in nums]

ts = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
out = '{OUT_DIR}/Flamingo_TEST_Bad_Ketchup_' + ts + '.csv'
with open(out, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['onset_ms'])
    for ms in onsets: w.writerow([ms])
print(f'ONSET_COUNT={{len(onsets)}}')
print(f'OUTPUT_FILE={{out}}')
"""

try:
    r = subprocess.run(
        [ENVS["flamingo"], "-c", flamingo_script],
        capture_output=True, text=True, timeout=600
    )
    out = r.stdout + r.stderr
    import re
    m = re.search(r'ONSET_COUNT=(\d+)', out)
    count = int(m.group(1)) if m else "?"
    results["Music-Flamingo"] = count
    print(f"✅ Music-Flamingo found {count} onsets")
    if r.returncode != 0:
        print(f"STDERR: {r.stderr[-500:]}")
except Exception as e:
    results["Music-Flamingo"] = f"ERROR: {e}"
    print(f"❌ Music-Flamingo failed: {e}")


# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("BENCHMARK SUMMARY — Bad Ketchup onset detection")
print("="*60)
for model, count in results.items():
    status = "✅" if isinstance(count, int) else "❌"
    print(f"  {status}  {model:<20} →  {count} onsets")
print("="*60)
