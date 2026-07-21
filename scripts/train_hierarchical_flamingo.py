#!/usr/bin/env python3
"""
train_hierarchical_flamingo.py
==============================
Trains Music-Flamingo as a Hierarchical Director.
Predicts ordered topological cluster tokens dynamically from 4-measure audio chunks.
"""
import os
import json
import torch
import librosa
import sqlite3
import numpy as np
import csv as csv_mod
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

try:
    import transformers
    from transformers import (
        AudioFlamingo3ForConditionalGeneration,
        AudioFlamingo3Processor,
    )
except ImportError:
    pass

# ── Paths ──
HF_MODEL_ID = "nvidia/music-flamingo-hf"
LOCAL_MODEL_PATH = "/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints/model_weights"
os.environ['HF_HOME'] = "/data/mg546924/llm_beatmap_generator/Music-Flamingo/checkpoints"
DB_PATH = "/data/mg546924/llm_beatmap_generator/pattern_finding_approach/processed_files.db"
TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"
OUTPUT_DIR = "/data/mg546924/models/music-flamingo-hierarchical-director"

NUM_EPOCHS = 5
LR = 2e-4
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_LENGTH = 512
MEASURES_PER_CHUNK = 4

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
        
        text = f"User: {prompt}\nAssistant: {clusters_str}"
        tok = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        prompt_text = f"User: {prompt}\nAssistant: "
        prompt_len = len(self.tokenizer(prompt_text).input_ids)
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        # Load Audio
        try:
            duration = sample["win_end"] - sample["win_start"]
            y, sr = librosa.load(sample["audio_path"], sr=16000, offset=sample["win_start"], duration=duration)
        except Exception:
            y = np.zeros(1)

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "audio": y}

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading processor...")
    processor = AudioFlamingo3Processor.from_pretrained(HF_MODEL_ID, trust_remote_code=True)
    
    with open(TOKENS_TXT, "r") as f:
        cluster_tokens = [line.strip() for line in f if line.strip()]
    processor.tokenizer.add_special_tokens({"additional_special_tokens": cluster_tokens})
    
    print("Loading model...")
    model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
        HF_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.gradient_checkpointing_enable()

    print("Injecting LoRA adapters (r=16)...")
    peft_config = LoraConfig(
        r=16, lora_alpha=32, target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    
    print("Loading dataset...")
    full_dataset = DynamicHierarchicalDataset(DB_PATH, processor.tokenizer, MAX_LENGTH, MEASURES_PER_CHUNK)
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    def collate_fn(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
        labels = torch.stack([b["labels"] for b in batch])
        
        audios = [b["audio"] for b in batch]
        
        # Audio Flamingo processes audio in the forward pass directly
        # Wait, if we use processor in the dataset, it's easier.
        # But we pass raw audio directly and use processor here.
        # However, to avoid memory issues, we'll let processor handle it.
        # I'll just use the raw arrays since `forward` expects it if not processed,
        # Actually Flamingo expects `audio_features`. 
        
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "audios": audios}

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    for name, param in model.named_parameters():
        if "lora" in name.lower() or "modules_to_save" in name.lower():
            param.requires_grad = True
        else:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    print(f"\nStarting Music-Flamingo Hierarchical Training\n")
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            try:
                # Processor processes the raw audio
                audio_inputs = processor(text=[""] * len(batch["audios"]), audio=batch["audios"], sampling_rate=16000, return_tensors="pt")
                audio_features = audio_inputs["input_features"].to(model.device) if "input_features" in audio_inputs else audio_inputs["audio_features"].to(model.device)
                
                outputs = model(
                    input_ids=batch["input_ids"].to(model.device), 
                    attention_mask=batch["attention_mask"].to(model.device), 
                    labels=batch["labels"].to(model.device),
                    audio_features=audio_features
                )
                loss = outputs.loss / GRAD_ACCUM
                loss.backward()
                
                if (batch_idx + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                epoch_loss += outputs.loss.item()
            except Exception as e:
                print(f"Error in training batch: {e}")
                optimizer.zero_grad()
                continue
                
        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                try:
                    audio_inputs = processor(text=[""] * len(batch["audios"]), audio=batch["audios"], sampling_rate=16000, return_tensors="pt")
                    audio_features = audio_inputs["input_features"].to(model.device) if "input_features" in audio_inputs else audio_inputs["audio_features"].to(model.device)
                    outputs = model(
                        input_ids=batch["input_ids"].to(model.device),
                        attention_mask=batch["attention_mask"].to(model.device),
                        labels=batch["labels"].to(model.device),
                        audio_features=audio_features
                    )
                    val_loss += outputs.loss.item()
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
                    
        print(f"Epoch {epoch+1} | Train: {epoch_loss/len(train_loader):.4f} | Val: {val_loss/len(val_loader):.4f}")

    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Complete!")

if __name__ == "__main__":
    main()
