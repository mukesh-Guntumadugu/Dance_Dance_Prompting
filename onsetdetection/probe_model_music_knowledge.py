"""
probe_model_music_knowledge.py
==============================
A simple diagnostic that asks each model three basic music questions
with NO audio input. Goal: verify the model is connected and understands
music vocabulary (onset, beat, tempo) before we trust onset outputs.

Usage (on cluster):
    python3 onsetdetection/probe_model_music_knowledge.py --model mumu
    python3 onsetdetection/probe_model_music_knowledge.py --model qwen
    python3 onsetdetection/probe_model_music_knowledge.py --model gemini
    python3 onsetdetection/probe_model_music_knowledge.py --model all
"""

import argparse
import sys
import os

QUESTIONS = [
    "What is a musical onset? Give a one-sentence definition.",
    "If a song has a tempo of 120 BPM, how many beats occur in 10 seconds?",
    "Name three ways to detect onsets in an audio signal.",
]

SEP = "─" * 70

def ask_mumu(question: str) -> str:
    sys.path.insert(0, "/data/mg546924/llm_beatmap_generator")
    from src.mumu_measure_interface import initialize_mumu_model
    import llama

    model, tokenizer = initialize_mumu_model()
    formatted = llama.utils.format_prompt(question)

    import torch
    with torch.no_grad():
        # No audio — pass a silent 1-second clip as a neutral placeholder
        silent_audio = [0.0] * 24000  # 1s of silence at 24kHz
        import numpy as np
        audio_np = np.array(silent_audio, dtype=np.float32)
        results = model.generate(
            prompts=[formatted],
            audios=audio_np,
            max_gen_len=256,
            temperature=0.2,
            top_p=0.9,
        )
    return results[0] if isinstance(results, list) else str(results)


def ask_qwen(question: str) -> str:
    sys.path.insert(0, "/data/mg546924/llm_beatmap_generator")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_path = "/data/mg546924/models/Qwen2-Audio-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    inputs = tokenizer(question, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, temperature=0.2, do_sample=True)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def ask_gemini(question: str) -> str:
    from dotenv import load_dotenv
    load_dotenv()
    import google.generativeai as genai
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(question)
    return response.text


MODEL_FUNCS = {
    "mumu":   ("MuMu-LLaMA",   ask_mumu),
    "qwen":   ("Qwen2-Audio",  ask_qwen),
    "gemini": ("Gemini Flash", ask_gemini),
}


def probe_model(name: str, label: str, ask_fn):
    print(f"\n{'='*70}")
    print(f"  MODEL: {label}")
    print(f"{'='*70}")
    for i, q in enumerate(QUESTIONS, 1):
        print(f"\n  Q{i}: {q}")
        print(f"  {SEP}")
        try:
            answer = ask_fn(q)
            print(f"  A:  {answer.strip()}")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
        print()


def main():
    parser = argparse.ArgumentParser(description="Probe model music knowledge")
    parser.add_argument("--model", choices=["mumu", "qwen", "gemini", "all"],
                        default="all", help="Which model to probe")
    args = parser.parse_args()

    targets = list(MODEL_FUNCS.items()) if args.model == "all" else [(args.model, MODEL_FUNCS[args.model])]

    print("\n🎵  Music Knowledge Probe — Checking model comprehension")
    print(f"    Questions: {len(QUESTIONS)}")
    print(f"    Models:    {[t[0] for t in targets]}")

    for name, (label, fn) in targets:
        probe_model(name, label, fn)

    print("\n✅  Probe complete.\n")


if __name__ == "__main__":
    main()
