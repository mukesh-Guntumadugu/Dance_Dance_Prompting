"""
train_qwen2_audio_grpo.py
=========================
GRPO (Group Relative Policy Optimization) training for Qwen2-Audio
to learn onset timestamp prediction from audio.

Key design:
  - Base model  : Qwen2-Audio-7B-Instruct (or SFT LoRA checkpoint)
  - Reward      : F1 score vs librosa ground-truth onsets (±50 ms tolerance)
  - Algorithm   : GRPO via TRL's GRPOTrainer
  - Adapter     : QLoRA (4-bit) to fit the 7B model on a single A6000 (48 GB)
  - Dataset     : sft_dataset_pixabay/dataset.jsonl (same as SFT)

Usage:
    python3 scripts/train_qwen2_audio_grpo.py
"""

import os
import re
import sys
import json
import torch
import librosa
import numpy as np
from typing import Any, Dict, List

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = "/data/mg546924/llm_beatmap_generator"
MODEL_ID     = os.environ.get(
    "BASE_MODEL",
    "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
)
# Optional: warm-start from an SFT checkpoint if it exists
SFT_CKPT     = os.environ.get(
    "SFT_CKPT",
    "/data/mg546924/models/qwen2-audio-lora-onsets"
)
DATASET_PATH = os.environ.get(
    "DATASET_OVERRIDE",
    f"{PROJECT_ROOT}/sft_dataset_pixabay/dataset.jsonl"
)
OUTPUT_DIR   = "/data/mg546924/models/qwen2-audio-grpo-onsets"
HF_CACHE     = "/data/mg546924/.cache/huggingface"

os.environ["HF_HOME"] = HF_CACHE
sys.path.insert(0, PROJECT_ROOT)

from src.onset_reward import onset_f1_reward

from datasets import load_dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import GRPOConfig, GRPOTrainer


# ── Helpers ────────────────────────────────────────────────────────────────────

def fix_audio_path(path: str) -> str:
    """Remap Mac local paths → cluster /data paths."""
    return path.replace(
        "/Users/mukeshguntumadugu/",
        "/data/mg546924/"
    )


def build_prompt(user_text: str) -> str:
    return (
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        f"<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n"
        f"{user_text}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


# ── Reward wrapper for GRPOTrainer ─────────────────────────────────────────────

def make_reward_fn(gt_onsets_map: Dict[str, List[float]]):
    """
    Returns a reward function compatible with GRPOTrainer's interface.
    GRPOTrainer calls reward_fn(prompts, completions, **kwargs).
    We embed the sample ID in the prompt so we can look up ground truth.
    """
    def reward_fn(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        rewards = []
        for prompt, completion in zip(prompts, completions):
            # Extract sample ID embedded in the prompt as a comment
            match = re.search(r"#ID:(\S+)", prompt)
            if match:
                sample_id = match.group(1)
                gt = gt_onsets_map.get(sample_id, [])
                r = onset_f1_reward(completion, gt)
            else:
                r = 0.0
            rewards.append(float(r))
        return rewards

    return reward_fn


# ── Dataset builder ────────────────────────────────────────────────────────────

def build_grpo_dataset(processor, dataset_path: str):
    """
    Loads the JSONL SFT dataset and converts it to a format GRPO expects:
    Each element has:
        - "prompt"      : tokenised text + audio inputs (dict)
        - "audio_array" : numpy array for the audio
        - "gt_onsets"   : list of float timestamps in seconds
        - "id"          : sample ID for reward lookup
    """
    raw = load_dataset("json", data_files={"train": dataset_path})["train"]

    samples = []
    skipped = 0
    for item in raw:
        try:
            msgs   = item["messages"]
            sample_id = item["id"]

            audio_url  = fix_audio_path(msgs[0]["content"][0]["audio_url"])
            user_text  = msgs[0]["content"][1]["text"]
            gt_str     = msgs[1]["content"][0]["text"]

            # Parse GT onsets from comma-separated seconds string
            gt_onsets = [float(x.strip()) for x in gt_str.split(",") if x.strip()]

            # Load audio
            y, sr = librosa.load(audio_url, sr=processor.feature_extractor.sampling_rate, mono=True)

            # Embed sample ID into prompt so reward_fn can look it up
            prompt_text = build_prompt(user_text) + f"#ID:{sample_id}"

            samples.append({
                "id":          sample_id,
                "prompt_text": prompt_text,
                "audio_array": y,
                "gt_onsets":   gt_onsets,
            })
        except Exception as e:
            skipped += 1
            continue

    print(f"Loaded {len(samples)} samples ({skipped} skipped) from {dataset_path}")
    return samples


# ── Collator ───────────────────────────────────────────────────────────────────

class GRPOAudioCollator:
    """Converts raw samples into processor-ready batches for GRPOTrainer."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts  = [f["prompt_text"] for f in features]
        audios = [f["audio_array"] for f in features]
        sr     = self.processor.feature_extractor.sampling_rate

        inputs = self.processor(
            text=texts,
            audio=audios,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        )
        return inputs


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Qwen2-Audio GRPO RL — Onset Detection Training")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load processor
    print(f"\n[1/5] Loading processor from {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True, fix_mistral_regex=True
    )

    # 2. Load & preprocess dataset
    print(f"\n[2/5] Building GRPO dataset from {DATASET_PATH}")
    samples = build_grpo_dataset(processor, DATASET_PATH)

    # Build GT lookup for reward function
    gt_map = {s["id"]: s["gt_onsets"] for s in samples}

    # Split train / eval (5% eval)
    split_idx = max(1, int(len(samples) * 0.95))
    train_samples = samples[:split_idx]
    eval_samples  = samples[split_idx:]
    print(f"  Train: {len(train_samples)} | Eval: {len(eval_samples)}")

    # 3. Load model in 4-bit QLoRA
    print(f"\n[3/5] Loading model in 4-bit QLoRA from {MODEL_ID}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # 4. GRPO config
    print("\n[4/5] Configuring GRPO trainer")
    grpo_cfg = GRPOConfig(
        output_dir=OUTPUT_DIR,

        # Generation settings (groups of rollouts per prompt)
        num_generations=4,           # 4 rollouts per prompt (memory-safe on A6000)
        max_new_tokens=256,          # enough for ~25 timestamps
        temperature=0.8,
        top_p=0.9,

        # KL penalty — prevents reward hacking / hallucination drift
        kl_coef=0.1,

        # Training
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-6,          # conservative for RL fine-tuning
        max_grad_norm=0.3,
        bf16=True,
        fp16=False,

        # Schedule (max_steps = ~10 hours)
        # Each step: 1 prompt × 4 rollouts × forward+backward → ~15-18s/step
        # 10 hours = 36000s → ~2000 steps is a safe budget
        max_steps=2000,
        warmup_steps=50,
        lr_scheduler_type="cosine",

        # Logging & saving
        logging_steps=10,
        save_steps=100,
        save_total_limit=5,
        report_to="none",
    )

    reward_fn = make_reward_fn(gt_map)

    trainer = GRPOTrainer(
        model=model,
        args=grpo_cfg,
        train_dataset=train_samples,
        eval_dataset=eval_samples,
        processing_class=processor,
        reward_funcs=reward_fn,
    )

    print("\n[5/5] Starting GRPO training (max 10 hours / 2000 steps)...")
    trainer.train()

    print("\nSaving final LoRA adapter...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Done. Model saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
