"""
validate_bpm_sensitivity.py
============================
Scientific validity test for LLM BPM estimation.

THEORY: A model that truly hears BPM must output proportionally scaled values
when the audio tempo is stretched/compressed. A hallucinating model will output
the same fixed value regardless of tempo perturbation.

The test perturbs a handful of songs at 3 speeds:
  - 0.5x  → BPM should halve
  - 1.0x  → BPM baseline
  - 1.5x  → BPM should be 1.5× baseline

Then computes a "Sensitivity Score" (0–100):
  100 = perfectly tracks tempo changes (real perception)
    0 = completely fixed / hallucinated

Usage:
    python3 onsetdetection/validate_bpm_sensitivity.py \\
        --model qwen \\
        --probe_songs 10 \\
        --batch_dir src/musicForBeatmap
"""

import argparse
import os
import sys
import csv
import tempfile
import numpy as np
import librosa
import soundfile as sf

ROOT = "/data/mg546924/llm_beatmap_generator"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

PROMPT = (
    "Listen to this audio clip carefully. "
    "What is the BPM (Beats Per Minute) of this audio? "
    "Output ONLY a single integer. No explanation."
)

SPEED_FACTORS = [0.5, 1.0, 1.5]


# ── Audio helpers ──────────────────────────────────────────────────────────────

def find_audio_file(folder_path: str) -> str:
    for f in sorted(os.listdir(folder_path)):
        if f.lower().endswith(('.ogg', '.mp3', '.wav')) and not f.startswith("._"):
            return os.path.join(folder_path, f)
    return None


def load_and_stretch(audio_path: str, speed: float, target_sr: int = 22050):
    """Load audio and time-stretch by 'speed' factor WITHOUT changing pitch."""
    y, sr = librosa.load(audio_path, sr=target_sr, duration=30.0)
    if speed != 1.0:
        # rate > 1 speeds up, rate < 1 slows down
        y = librosa.effects.time_stretch(y, rate=speed)
    return y, target_sr


def write_temp_wav(y: np.ndarray, sr: int) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y, sr)
    return tmp.name


def librosa_bpm(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    val = float(tempo[0]) if isinstance(tempo, (np.ndarray, list)) else float(tempo)
    return round(val, 2)


def parse_bpm_from_text(text: str) -> float:
    """Extract first number from model output text."""
    import re
    nums = re.findall(r'\b\d{2,3}(?:\.\d+)?\b', text)
    if nums:
        return float(nums[0])
    return -1.0


# ── Model inference helpers ────────────────────────────────────────────────────

def infer_qwen(y: np.ndarray, sr: int, processor, model) -> str:
    import torch
    messages = [
        {"role": "system", "content": "You are a concise musical assistant."},
        {"role": "user", "content": [
            {"type": "audio", "audio_url": "dummy"},
            {"type": "text", "text": PROMPT}
        ]}
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = processor(
        text=text, audios=[y],
        sampling_rate=sr, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=16,
            temperature=0.01, do_sample=False, use_cache=False
        )
    return processor.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def infer_mumu(y: np.ndarray, formatted_prompt, model) -> str:
    import torch
    import llama
    with torch.no_grad():
        with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
            out = model.generate(prompts=[formatted_prompt], audios=y, max_gen_len=16, temperature=0.01)
    return str(out[0]).strip() if isinstance(out, list) else str(out).strip()


def infer_deepresonance(tmp_path: str, model) -> str:
    inputs = {
        "inputs": ["<Audio>"],
        "instructions": [PROMPT],
        "mm_names": [["audio"]],
        "mm_paths": [[os.path.basename(tmp_path)]],
        "mm_root_path": os.path.dirname(tmp_path),
        "outputs": [""],
    }
    resp = model.predict(inputs, max_tgt_len=16, top_p=1.0, temperature=0.01, stops_id=[[835]])
    return str(resp[0]).strip() if isinstance(resp, list) and resp else str(resp).strip()


# ── Sensitivity Score ──────────────────────────────────────────────────────────

def compute_sensitivity_score(results_by_song: list) -> float:
    """
    For each song, compare:
        expected_0.5x = baseline * 0.5
        expected_1.5x = baseline * 1.5
    Score = how close the model tracks relative changes (0–100).
    """
    scores = []
    for entry in results_by_song:
        bpm_half  = entry.get("bpm_0.5x",  -1)
        bpm_base  = entry.get("bpm_1.0x",  -1)
        bpm_1half = entry.get("bpm_1.5x",  -1)

        if bpm_base <= 0:
            continue

        expected_half  = bpm_base * 0.5
        expected_1half = bpm_base * 1.5

        def rel_err(pred, expected):
            if pred <= 0 or expected <= 0:
                return 1.0
            return abs(pred - expected) / expected

        err_half  = rel_err(bpm_half,  expected_half)
        err_1half = rel_err(bpm_1half, expected_1half)
        song_score = max(0.0, 1.0 - (err_half + err_1half) / 2)
        scores.append(song_score)

    return round(np.mean(scores) * 100, 1) if scores else 0.0


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      choices=["qwen", "mumu", "deepresonance", "librosa"], required=True)
    parser.add_argument("--batch_dir",  required=True, help="Path to musicForBeatmap dataset")
    parser.add_argument("--probe_songs",type=int, default=10, help="Number of songs to test (default 10)")
    args = parser.parse_args()

    abs_batch_dir = os.path.abspath(args.batch_dir)

    # ── Collect song dirs ──
    song_dirs = []
    for root_dir, _, list_files in os.walk(abs_batch_dir, followlinks=True):
        if os.path.basename(root_dir).startswith("_"):
            continue
        if any(f.lower().endswith(('.ogg', '.mp3', '.wav')) and not f.startswith("._") for f in list_files):
            song_dirs.append(root_dir)

    song_dirs = sorted(set(song_dirs))[:args.probe_songs]
    print(f"\n🎯 Tempo Sensitivity Probe — Model: {args.model.upper()}")
    print(f"   Songs: {len(song_dirs)}  |  Speed factors: {SPEED_FACTORS}")
    print("─" * 68)

    # ── Load model once ──
    model_obj = None
    processor = None
    formatted_prompt = None

    if args.model == "qwen":
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
        import torch
        model_path = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model_obj = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True
        )
        target_sr = processor.feature_extractor.sampling_rate

    elif args.model == "mumu":
        from mumu_measure_interface import initialize_mumu_model
        import llama
        model_obj, _ = initialize_mumu_model()
        formatted_prompt = llama.utils.format_prompt(PROMPT)
        target_sr = 24000

    elif args.model == "deepresonance":
        from deepresonance_measure_interface import initialize_deepresonance_model
        model_obj = initialize_deepresonance_model()
        target_sr = 24000

    else:  # librosa — no model to load
        target_sr = 22050

    # ── Run perturbation tests ──
    results_by_song = []
    csv_rows = [["Song", "Librosa_BPM_1x",
                 "Speed_0.5x_Raw", "Speed_0.5x_BPM",
                 "Speed_1.0x_Raw", "Speed_1.0x_BPM",
                 "Speed_1.5x_Raw", "Speed_1.5x_BPM",
                 "Expected_0.5x", "Expected_1.5x",
                 "SensitivityScore"]]

    for idx, song_dir in enumerate(song_dirs):
        song_name = os.path.basename(song_dir)
        audio_path = find_audio_file(song_dir)
        if not audio_path:
            continue

        print(f"\n [{idx+1}/{len(song_dirs)}] {song_name}")

        # Librosa ground-truth BPM at 1× speed
        y_full, sr_full = librosa.load(audio_path, sr=22050, duration=30.0)
        librosa_bpm_1x = librosa_bpm(y_full, sr_full)
        print(f"   🎵 Librosa Ground Truth BPM: {librosa_bpm_1x}")

        entry = {"song": song_name}
        row = {"Song": song_name, "Librosa_BPM_1x": librosa_bpm_1x}

        for speed in SPEED_FACTORS:
            try:
                y, sr = load_and_stretch(audio_path, speed, target_sr=target_sr)
                raw_text = ""

                if args.model == "librosa":
                    bpm_val = librosa_bpm(y, sr)
                    raw_text = str(bpm_val)

                elif args.model == "qwen":
                    raw_text = infer_qwen(y, sr, processor, model_obj)
                    bpm_val = parse_bpm_from_text(raw_text)

                elif args.model == "mumu":
                    raw_text = infer_mumu(y, formatted_prompt, model_obj)
                    bpm_val = parse_bpm_from_text(raw_text)

                elif args.model == "deepresonance":
                    tmp_path = write_temp_wav(y, sr)
                    try:
                        raw_text = infer_deepresonance(tmp_path, model_obj)
                    finally:
                        os.unlink(tmp_path)
                    bpm_val = parse_bpm_from_text(raw_text)

                print(f"   Speed {speed}x → raw='{raw_text}'  parsed_bpm={bpm_val:.1f}  (expected {librosa_bpm_1x * speed:.1f})")
                entry[f"bpm_{speed}x"] = bpm_val
                row[f"Speed_{speed}x_Raw"] = raw_text
                row[f"Speed_{speed}x_BPM"] = bpm_val

            except Exception as e:
                print(f"   ⚠️  Speed {speed}x failed: {e}")
                entry[f"bpm_{speed}x"] = -1

        entry_score = compute_sensitivity_score([entry])
        entry["score"] = entry_score
        results_by_song.append(entry)

        row["Expected_0.5x"]     = round(librosa_bpm_1x * 0.5, 1)
        row["Expected_1.5x"]     = round(librosa_bpm_1x * 1.5, 1)
        row["SensitivityScore"]  = entry_score
        csv_rows.append([
            row["Song"], row["Librosa_BPM_1x"],
            row.get("Speed_0.5x_Raw", ""), row.get("Speed_0.5x_BPM", -1),
            row.get("Speed_1.0x_Raw", ""), row.get("Speed_1.0x_BPM", -1),
            row.get("Speed_1.5x_Raw", ""), row.get("Speed_1.5x_BPM", -1),
            row["Expected_0.5x"],  row["Expected_1.5x"],
            row["SensitivityScore"]
        ])

    # ── Final score ──
    overall_score = compute_sensitivity_score(results_by_song)
    print("\n" + "═" * 68)
    print(f"📊  OVERALL SENSITIVITY SCORE ({args.model.upper()}): {overall_score} / 100")
    print("─" * 68)
    print("  Interpretation:")
    print("   85–100 → Model genuinely tracks tempo changes (real audio perception)")
    print("   50–84  → Partial sensitivity (may combine audio + prior knowledge)")
    print("    0–49  → Mostly hallucinated / fixed values (not measuring audio)")
    print("═" * 68 + "\n")

    # ── Save CSV ──
    out_csv = os.path.join(ROOT, "onsetdetection", f"BPM_Sensitivity_{args.model.upper()}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(csv_rows)
    print(f"📁 Results saved: {out_csv}")


if __name__ == "__main__":
    main()
