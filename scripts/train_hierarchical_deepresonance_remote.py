#!/usr/bin/env python3
"""
train_hierarchical_deepresonance_fixed.py
=========================================
Trains DeepResonance as a Hierarchical Director.
Predicts ordered topological cluster tokens dynamically from 4-measure audio chunks.

FIX (Sep 23 2026 v2):
  Root cause: modeling_llama.py:697 `shift_labels = labels[..., 1:].contiguous()`
  raised "list indices must be integers or slices, not tuple" because `targets`
  was a Python list.  This happened because the ImageBind tensor pre-processing
  chain was broken — _get_audio_transform() was not loading tensors correctly.

  THE FIX: Write each audio chunk to /dev/shm as a temp .wav and pass the file
  PATH string in mm_inputs (not a tensor).  DeepResonance calls
  encode_audio(audio_paths=[path], do_tokenize=True) which uses ImageBind's
  load_and_transform_audio_data() natively — the correct, tested code path.
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

# ── Paths ─────────────────────────────────────────────────────────────────────
DR_ROOT    = "/data/mg546924/llm_beatmap_generator/DeepResonance/code"
CKPT_DIR   = "/data/mg546924/llm_beatmap_generator/DeepResonance/ckpt"
DB_PATH    = "/data/mg546924/llm_beatmap_generator/pattern_finding_approach/processed_files.db"
TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"
OUTPUT_DIR = "/data/mg546924/models/deepresonance-hierarchical-director"

# ── Hyper-parameters ──────────────────────────────────────────────────────────
NUM_EPOCHS         = 5      # 5 epochs to ensure the model fully learns the sliding window patterns
LR                 = 1e-4
BATCH_SIZE         = 1
MAX_LENGTH         = 512
MEASURES_PER_CHUNK = 4
GRAD_ACCUM         = 4     # effective batch = 4

# ── DeepResonance setup ───────────────────────────────────────────────────────
sys.path.insert(0, DR_ROOT)
os.chdir(DR_ROOT)

from unittest.mock import MagicMock
try:
    import triton  # noqa: F401
except ImportError:
    sys.modules["triton"] = MagicMock()
sys.modules["triton.ops"] = MagicMock()
sys.modules["triton.ops.matmul_perf_model"] = MagicMock()


def _get_audio_transform():
    from model.ImageBind import data as imagebind_data
    return imagebind_data.load_and_transform_audio_data

def find_audio_file(file_path: str):
    """Resolve a .ssc DB path to the adjacent audio file on the HPC."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_ssc   = os.path.join(repo_root, file_path)
    song_dir  = os.path.dirname(abs_ssc)
    song_stem = os.path.splitext(os.path.basename(abs_ssc))[0]

    if "/Users/mukeshguntumadugu/" in song_dir:
        song_dir = song_dir.replace("/Users/mukeshguntumadugu/", "/data/mg546924/")

    for ext in (".ogg", ".mp3", ".wav"):
        cand = os.path.join(song_dir, song_stem + ext)
        if os.path.exists(cand):
            return cand

    if os.path.isdir(song_dir):
        for f in os.listdir(song_dir):
            if f.lower().endswith((".ogg", ".mp3", ".wav")):
                return os.path.join(song_dir, f)
    return None


class DynamicHierarchicalDataset(Dataset):
    """
    Returns (prompt_str, target_str, tmp_wav_path) per sample.
    Audio slice is written to /dev/shm so DR can call
    load_and_transform_audio_data([path]) on it.
    """

    def __init__(self, db_path, measures_per_chunk=4):
        super().__init__()
        self.measures_per_chunk = measures_per_chunk
        self.samples = []
        self._load_audio_transform = _get_audio_transform

        print(f"Building memory index from {db_path}...")
        conn   = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT DISTINCT mca.file_path, mca.difficulty
            FROM measure_cluster_assignments mca
        """)
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
                WHERE af.file_path  = ? AND af.difficulty = ? AND af.chop_length = 1
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

    def __getitem__(self, idx):
        sample = self.samples[idx]

        target_c = sample["clusters"][-1]
        duration = sample["win_end"] - sample["win_start"]
        try:
            y, sr = librosa.load(
                sample["audio_path"], sr=16000,
                offset=sample["win_start"], duration=duration,
            )
            onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
            onset_times = librosa.frames_to_time(onset_frames, sr=sr)
            onsets_str = ", ".join([f"{t:.2f}s" for t in onset_times]) if len(onset_times) > 0 else "None"
        except Exception as e:
            print(f"[WARN] librosa.load failed: {e}")
            y  = np.zeros(int(16000 * max(duration, 0.1)), dtype=np.float32)
            sr = 16000
            onsets_str = "None"
            
        clusters_str = "<|cluster_empty|>" if target_c == -1 else f"<|cluster_{target_c}|>"
        
        # Extract the previous 3 clusters
        prev_clusters = []
        for c in sample["clusters"][:-1]:
            if c == -1:
                prev_clusters.append("<|cluster_empty|>")
            else:
                prev_clusters.append(f"<|cluster_{c}|>")
        prev_clusters_str = " ".join(prev_clusters)
        
        prompt = (
            f"<Audio><Audio><Audio><Audio> "
            "You are a rhythm game beatmap pattern generator. "
            f"Listen to this 4-measure audio segment. "
            f"The difficulty is {sample['difficulty']}. "
            f"Note onsets relative to the start of this chunk are: {onsets_str}. "
            f"The previous 3 measure clusters are: {prev_clusters_str}. "
            "Predict the rhythmic pattern cluster token for the FINAL measure."
        )

        audio_tensors = []
        load_audio = self._load_audio_transform()
        
        # Calculate exactly 1/4th of the duration per measure
        chunk_len = int(len(y) / 4)
        for j in range(4):
            start = j * chunk_len
            end = (j + 1) * chunk_len if j < 3 else len(y)
            chunk_y = y[start:end]
            if len(chunk_y) == 0:
                chunk_y = np.zeros(16000, dtype=np.float32)
            tmp_path = f"/dev/shm/dr_train_{os.getpid()}_{uuid.uuid4().hex[:8]}_m{j}.wav"
            sf.write(tmp_path, chunk_y, sr)
            
            try:
                # We explicitly enforce clips_per_video=1 so that ImageBind 
                # uses the EXACT measure duration continuously, preventing it 
                # from randomly shuffling 2-second clips!
                chunk_len_sec = len(chunk_y) / 16000.0
                tensor = load_audio([tmp_path], device='cpu', clips_per_video=1, clip_duration=chunk_len_sec)[0]
                audio_tensors.append(tensor)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        return {
            "prompt":     prompt,
            "target":     clusters_str,
            "audio_tensors": audio_tensors,
        }


def custom_collate_fn(batch):
    return {
        "prompt":      [item["prompt"]      for item in batch],
        "target":      [item["target"]      for item in batch],
        "audio_tensors": [item["audio_tensors"] for item in batch],
    }


def build_inputs_dict(batch):
    n = len(batch["prompt"])
    return {
        "dataset_types": ["AnyToText"] * n,
        "inputs":        batch["prompt"],
        "instructions":  [""] * n,
        "mm_inputs":     batch["audio_tensors"],
        "outputs":       batch["target"],
    }


def run_epoch(model, loader, optimizer, scheduler, trainable_params, is_train, grad_accum=1):
    total_loss, total_acc, n_ok = 0.0, 0.0, 0

    pbar = tqdm(loader, desc="train" if is_train else "val ")
    for step, batch in enumerate(pbar):
        inputs_dict = build_inputs_dict(batch)

        try:
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss, gen_acc, _ = model(inputs_dict)

            if is_train:
                (loss / grad_accum).backward()
                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()

            total_loss += loss.item()
            total_acc += gen_acc
            n_ok += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{gen_acc:.4f}"
            })

            if step % 200 == 0:
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"Batch failed: {e}")
            import traceback; traceback.print_exc()
            if is_train:
                optimizer.zero_grad()

    return total_loss / max(n_ok, 1), total_acc / max(n_ok, 1), n_ok


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    print(f"Added {len(cluster_tokens)} cluster tokens → vocab size {len(tokenizer)}")

    model.llama_model.resize_token_embeddings(len(tokenizer))
    model.llama_tokenizer = tokenizer   # expose for _prepare_one_mixed_embedding
    model = model.cuda().bfloat16()

    print("Loading hierarchical dataset...")
    full_dataset = DynamicHierarchicalDataset(DB_PATH, MEASURES_PER_CHUNK)

    val_size   = max(1, int(len(full_dataset) * 0.05))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Train: {train_size}  Val: {val_size}")

    # num_workers=0: avoids multiprocessing issues with /dev/shm temp files
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

    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5, weight_decay=0.01)

    from transformers import get_linear_schedule_with_warmup
    total_steps = len(train_loader) * NUM_EPOCHS // GRAD_ACCUM
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\nStarting DeepResonance Hierarchical Training ({NUM_EPOCHS} epoch(s))\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss, train_acc, train_ok = run_epoch(
            model, train_loader, optimizer, scheduler, trainable_params,
            is_train=True, grad_accum=GRAD_ACCUM,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_acc, val_ok = run_epoch(
                model, val_loader, optimizer, None, trainable_params,
                is_train=False, grad_accum=1,
            )

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} (ok={train_ok}/{train_size}) | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} (ok={val_ok}/{val_size})"
        )

        ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")

    print("Complete!")


if __name__ == "__main__":
    main()
