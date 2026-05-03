import json
from transformers import LlamaTokenizer
import os
import torch

LLAMA_DIR = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/LLaMA"
DATASET_JSONL = "/data/mg546924/llm_beatmap_generator/sft_dataset_5s_chunks/dataset.jsonl"

def check_dataset():
    print("Loading LLaMA tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_DIR)
    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")

    max_id = 0
    total_samples = 0
    with open(DATASET_JSONL, "r") as f:
        for line in f:
            item = json.loads(line)
            response = str(item["text_response"])
            # The prompt is also tokenized
            prompt = item["text_prompt"]
            text = f"Instruction: {prompt}\nAnswer: {response}"
            
            ids = tokenizer(text).input_ids
            if ids:
                max_id = max(max_id, max(ids))
            total_samples += 1

    print(f"Total samples checked: {total_samples}")
    print(f"Max ID found: {max_id}")
    if max_id >= vocab_size:
        print(f"CRITICAL: Found token ID {max_id} which is >= vocab size {vocab_size}!")
    else:
        print("All token IDs are within vocab range.")

if __name__ == "__main__":
    check_dataset()
