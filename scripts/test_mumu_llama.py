"""
Test script to verify if MuMu-LLaMA can be loaded from the cluster data drive.

Usage:
  python3 scripts/test_mumu_llama.py --ckpt /data/mg546924/llm_beatmap_generator/MuMu-LLaMA/ckpts/MuMu-LLaMA-MusicGen
"""

import sys
import argparse
import torch

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    # MuMu-LLaMA relies on specific internal imports depending on its branch/repo
    # This is a generic loader check for custom LLaMA architectures
except ImportError as e:
    print(f"❌ Dependency Error: {e}")
    print("Ensure you are in the correct MuMu python environment!")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser("Test MuMu-LLaMA Loading")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to MuMu-LLaMA checkpoint directory")
    args = parser.parse_args()

    print(f"🔄 Attempting to load MuMu-LLaMA from:\n   {args.ckpt}\n")

    if not torch.cuda.is_available():
        print("⚠️  Warning: CUDA is not available. Loading on CPU will be extremely slow/crash.")

    # In MuMu-LLaMA, the model is usually loaded via their custom pipeline,
    # but we can do a sanity check on the tokenizer and base weights first.
    try:
        print("1. Loading Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.ckpt, trust_remote_code=True)
        print("   ✅ Tokenizer loaded successfully!")

        print("\n2. Loading Model Weights (this may take a while)...")
        # Ensure we use float16 or bfloat16 to avoid OOM
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto" if torch.cuda.is_available() else None
        )
        print("   ✅ Model loaded successfully!")

        print("\n🎉 SUCCESS: The MuMu-LLaMA checkpoint and dependencies are fully operational!")

    except Exception as e:
        print("\n❌ FAILED TO LOAD MODEL")
        print("This is exactly why MuMu-LLaMA was abandoned earlier due to 'package interconnection' issues.")
        print("The error thrown by your environment is:")
        print("-" * 50)
        print(e)
        print("-" * 50)
        sys.exit(1)


if __name__ == "__main__":
    main()
