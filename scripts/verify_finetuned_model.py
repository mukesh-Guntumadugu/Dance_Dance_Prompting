#!/usr/bin/env python3
"""
verify_finetuned_model.py

Quick verification of the fine-tuned Qwen2-Audio LoRA model.
Runs the model on a few test songs, compares predicted onsets
against Librosa ground truth, and prints F1 score.

Usage (on cluster):
    python scripts/verify_finetuned_model.py \
        --audio_dir /data/mg546924/llm_beatmap_generator/pixabay_music \
        --lora_dir /data/mg546924/models/qwen2-audio-lora-onsets \
        --num_songs 3
"""

import argparse
import csv
import re
from pathlib import Path

import librosa
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

MODEL_ID = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"


def load_model(lora_dir: str):
    print(f"Loading base model: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    print(f"Loading LoRA adapter: {lora_dir}")
    model = PeftModel.from_pretrained(model, lora_dir)
    model.eval()
    return model, processor


def load_ground_truth(csv_path: Path) -> list[float]:
    """Load onset times (in seconds) from original_onsets_*.csv."""
    onsets = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            onsets.append(float(row["onset_ms"]) / 1000.0)
    return sorted(onsets)


CHUNK_DURATION = 20.0  # must match training


def predict_chunk(model, processor, y_chunk, start_offset: float) -> list[float]:
    """Run model on a single audio chunk, return absolute onset times."""
    prompt = (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
        "List the onsets in this audio segment as a comma-separated list of timestamps in seconds.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    inputs = processor(
        text=prompt,
        audio=[y_chunk],
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)

    text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = text.split("assistant\n")[-1].strip()

    onsets = []
    for token in re.split(r"[,\s]+", response):
        try:
            t = float(token)
            # Only keep onsets within the chunk window
            if 0 <= t <= CHUNK_DURATION:
                onsets.append(round(t + start_offset, 3))
        except ValueError:
            pass
    return onsets


def predict_onsets(model, processor, audio_path: str) -> list[float]:
    """
    Chunk audio into CHUNK_DURATION-second segments (matching training),
    run model on each chunk, aggregate onset timestamps.
    """
    import numpy as np
    sr = processor.feature_extractor.sampling_rate
    y, _ = librosa.load(audio_path, sr=sr)
    duration = len(y) / sr

    all_onsets = []
    for start in np.arange(0, duration, CHUNK_DURATION):
        end = min(start + CHUNK_DURATION, duration)
        if end - start < 2.0:
            continue
        start_frame = int(start * sr)
        end_frame = int(end * sr)
        y_chunk = y[start_frame:end_frame]
        chunk_onsets = predict_chunk(model, processor, y_chunk, start_offset=start)
        all_onsets.extend(chunk_onsets)

    return sorted(all_onsets)


def compute_f1(predicted: list[float], ground_truth: list[float], tolerance: float = 0.05) -> dict:
    """F1 score with a ±50ms tolerance window."""
    matched_pred = set()
    matched_gt = set()

    for i, pred in enumerate(predicted):
        for j, gt in enumerate(ground_truth):
            if j not in matched_gt and abs(pred - gt) <= tolerance:
                matched_pred.add(i)
                matched_gt.add(j)
                break

    tp = len(matched_pred)
    fp = len(predicted) - tp
    fn = len(ground_truth) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def main():
    parser = argparse.ArgumentParser(description="Verify fine-tuned Qwen2-Audio onset detection.")
    parser.add_argument("--audio_dir", type=str,
                        default="/data/mg546924/llm_beatmap_generator/pixabay_music")
    parser.add_argument("--lora_dir", type=str,
                        default="/data/mg546924/models/qwen2-audio-lora-onsets")
    parser.add_argument("--num_songs", type=int, default=3, help="Number of songs to test")
    parser.add_argument("--tolerance", type=float, default=0.05, help="Onset match tolerance in seconds")
    args = parser.parse_args()

    audio_dir = Path(args.audio_dir)
    audio_files = sorted(f for f in audio_dir.glob("*.mp3"))[:args.num_songs]

    if not audio_files:
        print(f"No MP3 files found in {audio_dir}")
        return

    model, processor = load_model(args.lora_dir)

    print(f"\n{'Song':<45} {'GT':>6} {'Pred':>6} {'P':>6} {'R':>6} {'F1':>6}")
    print("─" * 80)

    all_f1 = []
    for audio_path in audio_files:
        # Find matching ground truth CSV
        safe_stem = re.sub(r'[\\/:*?"<>|]', "_", audio_path.stem)
        csv_path = audio_dir / f"original_onsets_{safe_stem}.csv"
        if not csv_path.exists():
            print(f"  [SKIP] No onset CSV for {audio_path.name}")
            continue

        gt_onsets = load_ground_truth(csv_path)

        print(f"  Testing: {audio_path.name[:43]}", end="", flush=True)
        pred_onsets = predict_onsets(model, processor, str(audio_path))

        metrics = compute_f1(pred_onsets, gt_onsets, tolerance=args.tolerance)
        all_f1.append(metrics["f1"])

        print(f"\r  {audio_path.stem[:43]:<45} "
              f"{len(gt_onsets):>6} {len(pred_onsets):>6} "
              f"{metrics['precision']:>6.3f} {metrics['recall']:>6.3f} {metrics['f1']:>6.3f}")

    print("─" * 80)
    if all_f1:
        print(f"  Average F1: {sum(all_f1)/len(all_f1):.3f}  (±{args.tolerance*1000:.0f}ms tolerance)\n")


if __name__ == "__main__":
    main()
