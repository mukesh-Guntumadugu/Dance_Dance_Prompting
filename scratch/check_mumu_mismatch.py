import os
import sys
import torch
from transformers import LlamaTokenizer

# ── Paths ──
MUMU_ROOT = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/MuMu-LLaMA"
LLAMA_DIR = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/LLaMA"
MUMU_CKPT = "/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/MuMu-LLaMA-MusicGen/checkpoint.pth"

sys.path.insert(0, MUMU_ROOT)

def check_vocab_mismatch():
    from llama.mumu_llama import MuMu_LLaMA
    import argparse

    print("Loading tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(LLAMA_DIR)
    vocab_size = len(tokenizer)
    print(f"Tokenizer vocab size: {vocab_size}")

    model_args = argparse.Namespace(
        mert_path="m-a-p/MERT-v1-330M",
        vit_path="google/vit-base-patch16-224",
        vivit_path="google/vivit-b-16x2-kinetics400",
        music_decoder="musicgen",
        music_decoder_path="facebook/musicgen-small",
        max_words=512,
    )

    print("Loading model structure...")
    model = MuMu_LLaMA(
        llama_ckpt_dir=os.path.join(LLAMA_DIR, "7B"),
        llama_tokenizer=LLAMA_DIR,
        model_args=model_args,
        knn_dir="/data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts",
        stage=3,
    )
    
    emb_size = model.llama.tok_embeddings.weight.shape[0]
    print(f"Model embedding size: {emb_size}")

    if emb_size != vocab_size:
        print(f"CRITICAL MISMATCH: Model expects {emb_size} tokens, but tokenizer provides {vocab_size}!")
    else:
        print("Vocab sizes match.")

if __name__ == "__main__":
    check_vocab_mismatch()
