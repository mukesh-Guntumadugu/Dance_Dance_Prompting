#!/usr/bin/env python3
import os
import sys
import numpy as np

# Prevent accelerate from triggering DeepSpeed's buggy nvcc compiler check
sys.modules['deepspeed'] = None

import torch
import librosa
from datasets import load_dataset
from transformers import (
    Qwen2AudioForConditionalGeneration,
    AutoProcessor,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Dict, List, Any

# CONFIGURATION
MODEL_ID = "/data/mg546924/models/Qwen2-Audio-7B-Instruct" 
DB_PATH = "/data/mg546924/llm_beatmap_generator/pattern_finding_approach/processed_files.db"
TOKENS_TXT = "/data/mg546924/llm_beatmap_generator/scripts/cluster_to_patterns_tokens.txt"
OUTPUT_DIR = "/data/mg546924/models/qwen2-audio-hierarchical-director"
BLOCK_SIZE = 512
MEASURES_PER_CHUNK = 4       # Feed 4 measures at a time
MAX_SEQ_LENGTH = 512         # Max text token length


def load_cluster_tokens(tokens_txt_path):
    if not os.path.exists(tokens_txt_path):
        raise FileNotFoundError(f"Token list not found at {tokens_txt_path}.")
    with open(tokens_txt_path, "r") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens

def main():
    print(f"Loading processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True, fix_mistral_regex=True)
    
    # 1. Extend Tokenizer
    print("Extending tokenizer with cluster tokens...")
    cluster_tokens = load_cluster_tokens(TOKENS_TXT)
    before_len = len(processor.tokenizer)
    processor.tokenizer.add_special_tokens({"additional_special_tokens": cluster_tokens})
    after_len = len(processor.tokenizer)
    print(f"Tokenizer vocab size: {before_len} -> {after_len} (+{after_len - before_len})")

    # 2. Dynamic Dataset Loading
    print(f"Connecting to DB: {DB_PATH}")
    import sqlite3
    import torchaudio

    def find_audio_file(file_path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        abs_ssc = os.path.join(repo_root, file_path)
        song_dir = os.path.dirname(abs_ssc)
        song_stem = os.path.splitext(os.path.basename(abs_ssc))[0]
        
        # HPC Path translation if necessary
        if "/Users/mukeshguntumadugu/" in song_dir:
            song_dir = song_dir.replace("/Users/mukeshguntumadugu/", "/data/mg546924/")

        for ext in ['.ogg', '.mp3', '.wav']:
            candidate = os.path.join(song_dir, song_stem + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    class DynamicHierarchicalDataset(torch.utils.data.Dataset):
        def __init__(self, db_path, processor, measures_per_chunk=4):
            self.processor = processor
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
                        continue # Skip partial chunks
                    
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
            print(f"Built index of {len(self.samples)} dynamic {measures_per_chunk}-measure chunks.")

        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Efficiently load only the exact chunk we need!
            # Since torchaudio loads everything if we don't know the sr, we use librosa stream 
            # OR we just use librosa to load exactly the duration we want with offset.
            # Librosa offset is much safer for compressed audio (.mp3, .ogg) than torchaudio frame_offset.
            try:
                sr_target = self.processor.feature_extractor.sampling_rate
                duration = sample["win_end"] - sample["win_start"]
                y, sr = librosa.load(
                    sample["audio_path"],
                    sr=sr_target,
                    offset=sample["win_start"],
                    duration=duration
                )
            except Exception:
                # Return empty dummy arrays if load fails so collate doesn't crash, will be skipped
                y = np.zeros(1)
                
            cluster_tokens = " ".join(f"<|cluster_{c}|>" for c in sample["clusters"])
            
            prompt = (
                "You are a rhythm game beatmap pattern generator. "
                f"Listen to this audio segment which corresponds exactly to {self.measures_per_chunk} measure(s) in 4/4 time. "
                f"The difficulty is {sample['difficulty']}. "
                "Predict the ordered sequence of rhythmic pattern cluster tokens "
                "that best matches the audio's energy, density, and rhythm."
            )
            
            text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\nAudio 1: <|audio_bos|><|AUDIO|><|audio_eos|>\n{prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n{cluster_tokens}<|im_end|>\n"
            )
            
            inputs = self.processor(
                text=text,
                audio=[y],
                sampling_rate=self.processor.feature_extractor.sampling_rate,
                return_tensors="pt"
            )
            
            label = inputs["input_ids"].clone()
            try:
                response_ids = self.processor.tokenizer.encode(cluster_tokens + "<|im_end|>\n", add_special_tokens=False)
                resp_len = len(response_ids)
                label[0, :-resp_len] = -100
            except Exception:
                pass
                
            result = {
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                "labels": label[0],
            }
            
            if "audio_features" in inputs:
                result["audio_features"] = inputs["audio_features"][0]
            elif "input_features" in inputs:
                result["input_features"] = inputs["input_features"][0]
                
            if "feature_attention_mask" in inputs:
                result["feature_attention_mask"] = inputs["feature_attention_mask"][0]
                
            return result

    full_dataset = DynamicHierarchicalDataset(DB_PATH, processor, MEASURES_PER_CHUNK)
    val_size = int(len(full_dataset) * 0.05)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    print(f"Train dataset size: {train_size} | Validation dataset size: {val_size}")

    @dataclass
    class MultimodalDataCollator:
        processor: Any
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            if len(features) == 0:
                return {}
            input_ids = [torch.tensor(f["input_ids"], dtype=torch.long) for f in features]
            attention_mask = [torch.tensor(f["attention_mask"], dtype=torch.long) for f in features]
            labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
            
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
            
            batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
            # Use np.array() first — torch.tensor() on nested Python lists is extremely
            # slow for large audio feature arrays (e.g. [128, 500] takes seconds per sample).
            import numpy as np
            if "audio_features" in features[0]:
                batch["audio_features"] = torch.stack([torch.from_numpy(np.array(f["audio_features"], dtype=np.float16)) for f in features])
            elif "input_features" in features[0]:
                batch["input_features"] = torch.stack([torch.from_numpy(np.array(f["input_features"], dtype=np.float16)) for f in features])
                
            if "feature_attention_mask" in features[0]:
                batch["feature_attention_mask"] = torch.stack([torch.from_numpy(np.array(f["feature_attention_mask"], dtype=np.int64)) for f in features])
                
            return batch

    # 3. Model Loading (Pure bfloat16, load on CPU first to safely resize embeddings)
    print("Loading Base Model in bfloat16 on CPU...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16
    )
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # RESIZE EMBEDDINGS BEFORE PEFT AND BEFORE MOVING TO GPU
    print("Resizing token embeddings on CPU...")
    model.resize_token_embeddings(after_len, mean_resizing=False)
    
    print("Moving model to GPU...")
    model = model.to("cuda")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # 4. LoRA Adapter Config
    print("Injecting LoRA adapters...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"], 
        # IMPORTANT: embed_tokens and lm_head must be trained since we added new cluster tokens!
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Training Arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        per_device_eval_batch_size=1,
        optim="paged_adamw_32bit",
        logging_steps=1,           # Log every step so we can see per-step timing
        logging_strategy="steps",
        eval_strategy="epoch",
        save_strategy="no",
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        max_grad_norm=1.0,
        num_train_epochs=5,
        warmup_ratio=0.03,
        group_by_length=False,     # Disable: was hiding bad samples by grouping by length
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    # 6. Trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        data_collator=MultimodalDataCollator(processor),
    )
    
    print("Starting Training...")
    trainer.train()
    
    print(f"Saving final model adapter to {OUTPUT_DIR}...")
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()
