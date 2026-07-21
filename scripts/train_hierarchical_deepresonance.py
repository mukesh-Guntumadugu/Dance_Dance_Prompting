#!/usr/bin/env python3
"""
train_hierarchical_deepresonance.py
===================================
Trains DeepResonance as a Hierarchical Director.
Predicts ordered topological cluster tokens dynamically from 4-measure audio chunks.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import librosa
import sqlite3
import numpy as np
import csv as csv_mod
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

try:
    import transformers.utils.import_utils as _triu
    _triu.check_torch_load_is_safe = lambda: None
except Exception:
    pass

DR_ROOT = "/data/mg546924/llm_beatmap_generator/DeepResonance/code"
CKPT_DIR = "/data/mg546924/llm_beatmap_generator/DeepResonance/ckpt"
DB_PATH = "/data/mg546924/llm_beatmap_generator/pattern_finding_approach/processed_files.db"
TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"
OUTPUT_DIR = "/data/mg546924/models/deepresonance-hierarchical-director"

NUM_EPOCHS = 5
LR = 1e-4
BATCH_SIZE = 1
MAX_LENGTH = 512
MEASURES_PER_CHUNK = 4

sys.path.insert(0, DR_ROOT)
os.chdir(DR_ROOT)

from unittest.mock import MagicMock
try:
    import triton
except ImportError:
    sys.modules['triton'] = MagicMock()
sys.modules['triton.ops'] = MagicMock()
sys.modules['triton.ops.matmul_perf_model'] = MagicMock()

def find_audio_file(file_path):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    abs_ssc = os.path.join(repo_root, file_path)
    song_dir = os.path.dirname(abs_ssc)
    song_stem = os.path.splitext(os.path.basename(abs_ssc))[0]
    
    if "/Users/mukeshguntumadugu/" in song_dir:
        song_dir = song_dir.replace("/Users/mukeshguntumadugu/", "/data/mg546924/")

    for ext in ['.ogg', '.mp3', '.wav']:
        candidate = os.path.join(song_dir, song_stem + ext)
        if os.path.exists(candidate):
            return candidate
    return None

class DynamicHierarchicalDataset(Dataset):
    def __init__(self, db_path, tokenizer, max_length, measures_per_chunk=4):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.measures_per_chunk = measures_per_chunk
        self.samples = []
        
        print(f"Building memory index from {db_path}...")
        conn = sqlite3.connect(db_path, timeout=30)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DISTINCT af.file_path, af.difficulty
            FROM audio_features af
            JOIN measure_cluster_assignments mca
              ON af.file_path = mca.file_path AND af.difficulty = mca.difficulty AND af.run_id = mca.run_id
        """)
        songs = cursor.fetchall()
        
        for file_path, difficulty in songs:
            audio_path = find_audio_file(file_path)
            if not audio_path:
                continue
                
            cursor.execute("""
                SELECT af.start_time, af.end_time, mca.cluster_id
                FROM audio_features af
                JOIN measure_cluster_assignments mca
                  ON af.file_path = mca.file_path AND af.difficulty = mca.difficulty AND af.measure_idx = mca.measure_idx AND af.run_id = mca.run_id
                WHERE af.file_path = ? AND af.difficulty = ? AND mca.cluster_id != -1
                ORDER BY af.measure_idx ASC
            """, (file_path, difficulty))
            measures = cursor.fetchall()
            
            for i in range(0, len(measures), self.measures_per_chunk):
                chunk_measures = measures[i:i+self.measures_per_chunk]
                if len(chunk_measures) < self.measures_per_chunk:
                    continue
                
                win_start = chunk_measures[0][0]
                win_end = chunk_measures[-1][1]
                clusters = [m[2] for m in chunk_measures]
                
                self.samples.append({
                    "audio_path": audio_path,
                    "difficulty": difficulty,
                    "win_start": win_start,
                    "win_end": win_end,
                    "clusters": clusters
                })
        
        conn.close()
        print(f"Loaded {len(self.samples)} dynamic {measures_per_chunk}-measure chunks.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        clusters_str = " ".join(f"<|cluster_{c}|>" for c in sample["clusters"])
        
        prompt = (
            "You are a rhythm game beatmap pattern generator. "
            f"Listen to this audio segment which corresponds exactly to {self.measures_per_chunk} measure(s) in 4/4 time. "
            f"The difficulty is {sample['difficulty']}. "
            "Predict the ordered sequence of rhythmic pattern cluster tokens "
            "that best matches the audio's energy, density, and rhythm."
        )
        
        text = f"### Human: {prompt}\n### Assistant: {clusters_str}"
        tokens = self.tokenizer(text, return_tensors="pt", max_length=self.max_length,
                                 truncation=True, padding="max_length")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        prompt_text = f"### Human: {prompt}\n### Assistant: "
        prompt_len = len(self.tokenizer(prompt_text).input_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "audio_path": sample["audio_path"]
        }

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    from config import load_config
    from model.deepresonance import DeepResonanceModel
    from transformers import LlamaTokenizer

    args = {
        'model': 'deepresonance',
        'stage': 2,
        'mode': 'train',
        'max_length': MAX_LENGTH,
        'max_output_length': MAX_LENGTH,
        'ckpt_path': os.path.join(CKPT_DIR, 'deepresonance_alpha_delta_ckpt'),
        'pretrained_ckpt_path': os.path.join(CKPT_DIR, 'pretrained_ckpt'),
    }
    config = load_config(args)
    args.update(config)
    args['max_length'] = MAX_LENGTH

    print("Loading DeepResonance model...")
    model = DeepResonanceModel(**args)
    
    delta_path = os.path.join(args['ckpt_path'], 'pytorch_model.pt')
    if os.path.exists(delta_path):
        delta_ckpt = torch.load(delta_path, map_location='cpu')
        model.load_state_dict(delta_ckpt, strict=False)

    vicuna_path = os.path.join(CKPT_DIR, 'pretrained_ckpt', 'vicuna-7b-v1.1')
    tokenizer = LlamaTokenizer.from_pretrained(vicuna_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    with open(TOKENS_TXT, "r") as f:
        cluster_tokens = [line.strip() for line in f if line.strip()]
    tokenizer.add_special_tokens({"additional_special_tokens": cluster_tokens})
    
    print("Resizing token embeddings...")
    model.llama_model.resize_token_embeddings(len(tokenizer))

    model = model.cuda().bfloat16()
    model.train()

    print("Loading hierarchical dataset...")
    full_dataset = DynamicHierarchicalDataset(DB_PATH, tokenizer, MAX_LENGTH, MEASURES_PER_CHUNK)

    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    for name, param in model.named_parameters():
        if "lora" in name.lower() or "delta" in name.lower() or "embed_tokens" in name.lower() or "lm_head" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=LR, weight_decay=0.05)

    print(f"\nStarting DeepResonance Hierarchical Training\n")

    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()

            try:
                with torch.cuda.amp.autocast():
                    outputs = model.llama_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs.loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                if (batch_idx + 1) % 500 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                continue

        avg_train_loss = epoch_loss / max(num_batches, 1)

        model.eval()
        val_loss, val_batches = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    with torch.cuda.amp.autocast():
                        outputs = model.llama_model(
                            input_ids=batch["input_ids"].cuda(),
                            attention_mask=batch["attention_mask"].cuda(),
                            labels=batch["labels"].cuda(),
                        )
                        val_loss += outputs.loss.item()
                        val_batches += 1
                except:
                    continue
        
        avg_val_loss = val_loss / max(val_batches, 1)
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if epoch + 1 == NUM_EPOCHS:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch+1}.pt")
            torch.save(model.state_dict(), ckpt_path)

    print("Complete!")

if __name__ == "__main__":
    main()
