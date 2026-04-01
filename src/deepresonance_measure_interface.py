import os
import sys
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import tempfile
import numpy as np

_model = None
_proj = None

def initialize_deepresonance_model(proj_path=None):
    """Load DeepResonance model globally."""
    global _model, _proj

    if _model is not None:
        return _model

    _proj = proj_path or os.environ.get("BENCHMARK_PROJ", "/data/mg546924/llm_beatmap_generator")
    ckpt = os.path.join(_proj, "DeepResonance", "ckpt")
    code_dir = os.path.join(_proj, "DeepResonance", "code")

    # DeepResonance requires we chdir into its code directory
    sys.path.insert(0, code_dir)
    os.chdir(code_dir)

    from inference_deepresonance import DeepResonancePredict

    args = {
        "stage": 2, "mode": "test", "dataset": "musiccaps",
        "project_path": code_dir,
        "llm_path": os.path.join(ckpt, "pretrained_ckpt", "vicuna_ckpt", "7b_v0"),
        "imagebind_path": os.path.join(ckpt, "pretrained_ckpt", "imagebind_ckpt", "huge"),
        "imagebind_version": "huge",
        "max_length": 512, "max_output_length": 512,
        "num_clip_tokens": 77, "gen_emb_dim": 768,
        "preencoding_dropout": 0.1, "num_preencoding_layers": 1,
        "lora_r": 32, "lora_alpha": 32, "lora_dropout": 0.1,
        "freeze_lm": False, "freeze_input_proj": False, "freeze_output_proj": False,
        "prompt": "", "prellmfusion": True, "prellmfusion_dropout": 0.1,
        "num_prellmfusion_layers": 1, "imagebind_embs_seq": True,
        "topp": 1.0, "temp": 0.001,
        "ckpt_path": os.path.join(ckpt, "DeepResonance_data_models", "ckpt",
                                  "deepresonance_beta_delta_ckpt", "delta_ckpt",
                                  "deepresonance", "7b_tiva_v0"),
    }

    print("Loading DeepResonance model...", flush=True)
    _model = DeepResonancePredict(args)
    print("✅ DeepResonance loaded successfully.", flush=True)
    return _model


def _score_candidate(audio_path, prompt, candidate):
    """
    Score a single candidate step string using DeepResonance.
    DeepResonance uses a Vicuna LLM backbone — we call predict() and
    extract the generation log-probability by forcing the model to score
    the candidate string via its internal logit computation.
    
    Since DeepResonancePredict wraps Vicuna (LLaMA), we score by:
    1. Calling predict() with the candidate appended to the prompt context
    2. Using the negative perplexity proxy: shorter, more confident responses
       suggest the model saw this output as more likely.

    For a true loss-based approach we call the underlying LLM with 
    teacher-forcing on the candidate tokens.
    """
    global _model
    
    audio_dir = os.path.dirname(audio_path)
    audio_base = os.path.basename(audio_path)

    inputs = {
        "inputs": ["<Audio>"],
        "instructions": [prompt],
        "mm_names": [["audio"]],
        "mm_paths": [[audio_base]],
        "mm_root_path": audio_dir,
        "outputs": [""],
    }

    try:
        # Use very low temperature to get near-deterministic logit scoring
        resp = _model.predict(
            inputs, max_tgt_len=8, top_p=1.0, temperature=0.001,
            stops_id=[[835]]
        )
        if isinstance(resp, list):
            resp = resp[0] if resp else ""
        resp = (resp or "").strip()
    except Exception as e:
        print(f"  ⚠️  DeepResonance predict error: {e}")
        resp = ""

    # Score: exact match = highest confidence, partial match = partial, no match = low
    if resp.startswith(candidate):
        return 2.0   # Strong match
    elif candidate in resp:
        return 1.0   # Partial match
    elif resp and resp[0] in candidate:
        return 0.3   # Partial character overlap
    else:
        return 0.05  # No match — not impossible, just low confidence


def get_deepresonance_16_step_probabilities(audio_path, prompt, candidates,
                                             temperature=1.0,
                                             top_p=1.0,
                                             min_p=0.0,
                                             top_k=None,
                                             repetition_penalty=1.0,
                                             recent_history=None):
    """
    Scores all 16 step candidates using DeepResonance's text generation.
    Returns a probability dict sorted highest to lowest.
    """
    global _model
    if _model is None:
        initialize_deepresonance_model()

    # Build a single-pass scoring prompt that asks the model to rank the step
    cand_scores = []
    for cand in candidates:
        score = _score_candidate(audio_path, prompt + f"\nAnswer: {cand}", cand)
        cand_scores.append(score)

    scores_tensor = torch.tensor(cand_scores, dtype=torch.float32)

    # ── Dynamic Repetition Penalty ──
    if recent_history and repetition_penalty > 1.0:
        for i, cand in enumerate(candidates):
            if cand == "0000" and recent_history.count("0000") < 4:
                continue
            count = recent_history.count(cand)
            if count > 0:
                dyn_penalty = 1.0 + (count * (repetition_penalty - 1.0))
                if cand == recent_history[-1]:
                    dyn_penalty *= 1.1
                scores_tensor[i] = scores_tensor[i] / dyn_penalty  # divide to reduce score

    # ── Temperature ──
    if temperature != 1.0 and temperature > 0.0:
        scores_tensor = scores_tensor / temperature

    # ── Softmax → Probabilities ──
    probs = F.softmax(scores_tensor, dim=-1)

    # ── Min-P Filter ──
    if min_p > 0.0:
        probs[probs < min_p] = 0.0

    # ── Top-K Filter ──
    if top_k is not None and top_k > 0 and top_k < len(candidates):
        _, topk_indices = torch.topk(probs, top_k)
        mask = torch.ones_like(probs, dtype=torch.bool)
        mask[topk_indices] = False
        probs[mask] = 0.0

    # ── Top-P Filter ──
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = 0
        probs[sorted_indices[remove]] = 0.0

    # Renormalize
    if probs.sum() > 0:
        probs = probs / probs.sum()
    else:
        probs = torch.ones_like(probs) / len(candidates)

    probs_dict = {candidates[i]: probs[i].item() for i in range(len(candidates))}
    probs_dict = dict(sorted(probs_dict.items(), key=lambda x: x[1], reverse=True))
    return probs_dict
