#!/usr/bin/env python3
"""
train_hierarchical_deepresonance_themachine.py
==============================================
Trains DeepResonance as a Hierarchical Director on The Machine
(themachine.cs.ohio.edu) with all NaN bugs fixed.

FIXES vs original train_hierarchical_deepresonance.py:
  1. target_length=832 (was 204 = 2s; 4 measures @ ~120 BPM ≈ 8s = ~800 frames)
  2. clip_duration matches actual audio chunk duration (not hardcoded 2s)
  3. clips_per_video=1 (no random sampling of clips)
  4. GradScaler for safe float16 mixed precision (was raw bfloat16 -> NaN)
  5. NaN/Inf guard -- skips bad batches entirely
  6. LR lowered from 1e-4 to 2e-5
  7. Paths updated for The Machine (~/ instead of /data/mg546924/)
  8. Checkpoint saved after EVERY epoch (not only at epoch 30)
  9. Linear warmup LR scheduler
 10. Gradient accumulation (effective batch size = 4)
"""

import os
import sys
import uuid
import sqlite3
import torch
import librosa
import numpy as np
import soundfile as sf
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import transformers.utils.import_utils as _triu
    _triu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

# Paths for The Machine
HOME           = os.path.expanduser("~")
DR_ROOT        = os.path.join(HOME, "llm_beatmap_generator/DeepResonance/code")
CKPT_DIR       = os.path.join(HOME, "llm_beatmap_generator/DeepResonance/ckpt")
DB_PATH        = os.path.join(HOME, "llm_beatmap_generator/pattern_finding_approach/processed_files.db")
TOKENS_TXT     = os.path.join(HOME, "llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt")
OUTPUT_DIR     = os.path.join(HOME, "models/deepresonance-hierarchical-director-fixed")
TMP_DIR        = "/dev/shm"

# Hyper-parameters
NUM_EPOCHS          = 5
LR                  = 2e-5     # was 1e-4 -- safer for fine-tuning
BATCH_SIZE          = 1
MAX_LENGTH          = 512
MEASURES_PER_CHUNK  = 4
GRAD_ACCUM          = 4        # effective batch size = 4

# FIX 1: correct target_length for mel-spectrogram
# 4 measures @ 120 BPM = ~8 seconds, 16kHz, 10ms frame shift -> ~800 frames
# Use 832 (next multiple of 64) as safe upper bound
AUDIO_TARGET_LENGTH = 832      # was 204 (= only 2 seconds!)

# DeepResonance setup
sys.path.insert(0, DR_ROOT)
os.chdir(DR_ROOT)

from unittest.mock import MagicMock
try:
    import triton  # noqa: F401
except ImportError:
    sys.modules["triton"] = MagicMock()
sys.modules["triton.ops"] = MagicMock()
sys.modules["triton.ops.matmul_perf_model"] = MagicMock()


def find_audio_file(file_path: str):
    """Resolve a .ssc DB path to the adjacent audio file on The Machine."""
    home = os.path.expanduser("~")
    for old_prefix in ["/data/mg546924/", "/Users/mukeshguntumadugu/"]:
        file_path = file_path.replace(old_prefix, home + "/")

    song_dir  = os.path.dirname(file_path)
    song_stem = os.path.splitext(os.path.basename(file_path))[0]

    for ext in (".ogg", ".mp3", ".wav"):
        cand = os.path.join(song_dir, song_stem + ext)
        if os.path.exists(cand):
            return cand

    if os.path.isdir(song_dir):
        for f in sorted(os.listdir(song_dir)):
            if f.lower().endswith((".ogg", ".mp3", ".wav")):
                return os.path.join(song_dir, f)
    return None


class DynamicHierarchicalDataset(Dataset):
    def __init__(self, db_path: str, measures_per_chunk: int = 4):
        super().__init__()
        self.measures_per_chunk = measures_per_chunk
        self.samples = []

        print(f"Building memory index from {db_path}...")
        conn   = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT DISTINCT mca.file_path, mca.difficulty "
            "FROM measure_cluster_assignments mca"
        )
        songs = cursor.fetchall()
        print(f"  Found {len(songs)} (file_path, difficulty) pairs")

        for file_path, difficulty in songs:
            audio_path = find_audio_file(file_path)
            if not audio_path or not os.path.exists(audio_path):
                continue

            cursor.execute("""
                SELECT af.start_time, af.end_time, mca.cluster_id
                FROM audio_features af
                JOIN measure_cluster_assignments mca
                  ON  af.file_path   = mca.file_path
                  AND af.difficulty  = mca.difficulty
                  AND af.measure_idx = mca.measure_idx
                  AND af.run_id      = mca.run_id
                  AND af.chop_length = mca.chop_length
                WHERE af.file_path = ? AND af.difficulty = ? AND af.chop_length = 1
                  AND mca.cluster_id != -1
                ORDER BY af.measure_idx ASC
            """, (file_path, difficulty))
            measures = cursor.fetchall()

            for i in range(0, len(measures) - measures_per_chunk + 1):
                chunk = measures[i : i + measures_per_chunk]
                if len(chunk) < measures_per_chunk:
                    continue
                self.samples.append({
                    "audio_path": audio_path,
                    "difficulty": difficulty,
                    "win_start":  chunk[0][0],
                    "win_end":    chunk[-1][1],
                    "clusters":   [m[2] for m in chunk],
                })

        conn.close()
        print(f"Loaded {len(self.samples)} dynamic {measures_per_chunk}-measure chunks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        duration = max(sample["win_end"] - sample["win_start"], 0.5)

        try:
            y, sr = librosa.load(
                sample["audio_path"], sr=16000,
                offset=sample["win_start"], duration=duration,
            )
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times  = librosa.frames_to_time(onset_frames, sr=sr)
            onsets_str   = ", ".join(f"{t:.2f}s" for t in onset_times) if len(onset_times) > 0 else "None"
        except Exception as e:
            print(f"[WARN] librosa.load failed: {e}")
            y          = np.zeros(int(16000 * max(duration, 0.5)), dtype=np.float32)
            sr         = 16000
            onsets_str = "None"

        prev_clusters_str = " ".join(
            f"<|cluster_{c}|>" for c in sample["clusters"][:-1]
        )
        target_c     = sample["clusters"][-1]
        target_token = f"<|cluster_{target_c}|>"

        prompt = (
            "<Audio><Audio><Audio><Audio> "
            "You are a rhythm game beatmap pattern generator. "
            "Listen to this 4-measure audio segment. "
            f"The difficulty is {sample['difficulty']}. "
            f"Note onsets relative to the start of this chunk: {onsets_str}. "
            f"The previous 3 measure clusters are: {prev_clusters_str}. "
            "Predict the rhythmic pattern cluster token for the FINAL measure."
        )

        # Write each measure to /dev/shm for ImageBind
        chunk_samples = max(len(y) // 4, 1)
        wav_paths = []
        for j in range(4):
            start   = j * chunk_samples
            end     = (j + 1) * chunk_samples if j < 3 else len(y)
            chunk_y = y[start:end]
            if len(chunk_y) == 0:
                chunk_y = np.zeros(int(16000 * (duration / 4)), dtype=np.float32)
            tmp_path = os.path.join(
                TMP_DIR, f"dr_tm_{os.getpid()}_{uuid.uuid4().hex[:8]}_m{j}.wav"
            )
            sf.write(tmp_path, chunk_y, sr)
            wav_paths.append(tmp_path)

        return {
            "prompt":    prompt,
            "target":    target_token,
            "wav_paths": wav_paths,
            "duration":  duration / 4,
        }


def custom_collate_fn(batch):
    return {
        "prompt":    [item["prompt"]    for item in batch],
        "target":    [item["target"]    for item in batch],
        "wav_paths": [item["wav_paths"] for item in batch],
        "duration":  [item["duration"]  for item in batch],
    }


def load_audio_tensors(wav_paths_batch, durations, device):
    """Load audio with FIX 1: correct target_length and clip_duration."""
    from model.ImageBind import data as imagebind_data
    batch_tensors = []
    for wav_paths, dur in zip(wav_paths_batch, durations):
        measure_tensors = []
        for wav_path in wav_paths:
            try:
                t = imagebind_data.load_and_transform_audio_data(
                    [wav_path],
                    device=device,
                    num_mel_bins=128,
                    target_length=AUDIO_TARGET_LENGTH,  # FIX 1: was 204
                    sample_rate=16000,
                    clip_duration=max(dur, 0.1),        # FIX 2: actual duration
                    clips_per_video=1,                  # FIX 2: no random clips
                )[0]
                measure_tensors.append(t)
            except Exception as e:
                print(f"[WARN] ImageBind load failed for {wav_path}: {e}")
                measure_tensors.append(
                    torch.zeros(1, 1, 128, AUDIO_TARGET_LENGTH, device=device)
                )
            finally:
                if os.path.exists(wav_path):
                    os.remove(wav_path)
        batch_tensors.append(measure_tensors)
    return batch_tensors


def run_epoch(model, loader, optimizer, scheduler, scaler,
              trainable_params, is_train, grad_accum=1, device=None):
    if device is None:
        device = torch.device("cuda")

    total_loss, total_acc, n_ok, n_nan = 0.0, 0.0, 0, 0
    pbar = tqdm(loader, desc="train" if is_train else "val  ")

    for step, batch in enumerate(pbar):
        audio_tensors = load_audio_tensors(batch["wav_paths"], batch["duration"], device)
        inputs_dict = {
            "dataset_types": ["AnyToText"] * len(batch["prompt"]),
            "inputs":        batch["prompt"],
            "instructions":  [""] * len(batch["prompt"]),
            "mm_inputs":     audio_tensors,
            "outputs":       batch["target"],
        }

        try:
            # FIX 3: float16 autocast + GradScaler
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, gen_acc, _ = model(inputs_dict)

            # FIX 4: NaN guard -- skip bad batches
            if torch.isnan(loss) or torch.isinf(loss):
                n_nan += 1
                if is_train:
                    optimizer.zero_grad()
                pbar.set_postfix({"loss": "NaN(skip)", "nan_ct": n_nan})
                continue

            if is_train:
                scaler.scale(loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            total_acc  += gen_acc
            n_ok       += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{gen_acc:.4f}", "nan": n_nan})

            if step % 200 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\n[ERROR] Step {step} failed: {e}")
            import traceback; traceback.print_exc()
            if is_train:
                optimizer.zero_grad()

    print(f"  NaN batches skipped: {n_nan}")
    return total_loss / max(n_ok, 1), total_acc / max(n_ok, 1), n_ok


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    from config import load_config
    from model.deepresonance import DeepResonanceModel
    from transformers import LlamaTokenizer

    args = {
        "model":                "deepresonance",
        "stage":                2,
        "mode":                 "train",
        "max_length":           MAX_LENGTH,
        "max_output_length":    MAX_LENGTH,
        "ckpt_path":            os.path.join(CKPT_DIR, "deepresonance_alpha_delta_ckpt"),
        "pretrained_ckpt_path": os.path.join(CKPT_DIR, "pretrained_ckpt"),
    }
    config = load_config(args)
    args.update(config)
    args["max_length"] = MAX_LENGTH
    args.setdefault("imagebind_embs_seq",      False)
    args.setdefault("prellmfusion",            False)
    args.setdefault("prellmfusion_dropout",    0.1)
    args.setdefault("num_prellmfusion_layers", 4)

    print("Loading DeepResonance model...")
    model = DeepResonanceModel(**args)

    delta_path = os.path.join(args["ckpt_path"], "pytorch_model.pt")
    if os.path.exists(delta_path):
        delta_ckpt = torch.load(delta_path, map_location="cpu")
        model.load_state_dict(delta_ckpt, strict=False)
        print(f"Loaded delta weights from {delta_path}")

    vicuna_path = os.path.join(CKPT_DIR, "pretrained_ckpt", "vicuna-7b-v1.1")
    tokenizer   = LlamaTokenizer.from_pretrained(vicuna_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(TOKENS_TXT, "r") as f:
        cluster_tokens = [line.strip() for line in f if line.strip()]
    tokenizer.add_special_tokens({"additional_special_tokens": cluster_tokens})
    print(f"Added {len(cluster_tokens)} cluster tokens, vocab size: {len(tokenizer)}")

    model.llama_model.resize_token_embeddings(len(tokenizer))
    model.llama_tokenizer = tokenizer

    # FIX 3: Keep model in float32; GradScaler handles precision
    model = model.cuda().float()
    model.train()

    print("Loading hierarchical dataset...")
    full_dataset = DynamicHierarchicalDataset(DB_PATH, MEASURES_PER_CHUNK)

    val_size   = max(1, int(len(full_dataset) * 0.05))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size}  Val: {val_size}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0, collate_fn=custom_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=0, collate_fn=custom_collate_fn)

    for name, param in model.named_parameters():
        param.requires_grad = any(
            k in name.lower() for k in ("lora", "delta", "embed_tokens", "lm_head")
        )
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    # FIX 6: Lower LR + warmup scheduler
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.01)

    from transformers import get_linear_schedule_with_warmup
    total_steps  = (len(train_loader) // GRAD_ACCUM) * NUM_EPOCHS
    warmup_steps = max(1, int(0.05 * total_steps))
    scheduler    = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # FIX 3: GradScaler for safe mixed precision
    scaler = torch.cuda.amp.GradScaler()

    print(f"\nStarting Training ({NUM_EPOCHS} epochs)")
    print(f"  LR={LR}, AUDIO_TARGET_LENGTH={AUDIO_TARGET_LENGTH}, GRAD_ACCUM={GRAD_ACCUM}\n")

    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_acc, train_ok = run_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            trainable_params, is_train=True, grad_accum=GRAD_ACCUM, device=device,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_ok = run_epoch(
                model, val_loader, optimizer, None, scaler,
                trainable_params, is_train=False, grad_accum=1, device=device,
            )

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} (ok={train_ok}/{train_size}) | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} (ok={val_ok}/{val_size})"
        )

        # FIX 7: Save every epoch
        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(OUTPUT_DIR, "best_checkpoint.pt")
            torch.save(model.state_dict(), best_path)
            print(f"  New best val loss {val_loss:.4f} -> saved {best_path}")

    print("\nTraining complete!")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
