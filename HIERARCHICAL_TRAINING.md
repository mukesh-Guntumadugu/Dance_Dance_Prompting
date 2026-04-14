# 🎵 Hierarchical Director–Actor Beatmap Training

This document explains the current training approach running on the HPC cluster.

---

## The Core Idea

Instead of generating beatmap notes one by one, we use a **two-stage pipeline**:

```
Audio Input
    │
    ▼
┌─────────────────────────────────────┐
│      DIRECTOR  ← training this now  │
│                                     │
│  MuMu-LLaMA + MERT audio encoder   │
│                                     │
│  Listens to music → decides which   │
│  rhythmic PATTERN fits each section │
│                                     │
│  Output: cluster token sequence     │
│  e.g.  [CLU_0042] [CLU_1337] ...   │
└─────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│      ACTOR  (pre-built lookup)      │
│                                     │
│  cluster token → 192×4 note grid   │
│                                     │
│  Output: full playable beatmap      │
└─────────────────────────────────────┘
```

**Why this is better than direct generation:**

| Direct (old way) | Hierarchical (this way) |
|-----------------|------------------------|
| Predict every note individually | Predict one cluster per measure |
| 192 decisions per measure | 1 decision per measure |
| No musical phrasing structure | Natural musical structure emerges |
| Audio conditioning is hard | Audio directly drives cluster choice |
| Slow inference | Fast — cluster lookup is instant |

---

## Step 1 — Pattern Clustering (`pattern_finding_approach/`)

Every measure from every beatmap was converted into a **192×4 binary matrix**  
(192 time subdivisions × 4 arrow lanes).

Then we applied:
1. **PCA** — compresses the 768-dimensional measure vector
2. **HDBSCAN** — density-based clustering, finds natural rhythm groupings
3. **UMAP** — 2D visualization to inspect the clusters

**Result: 2189 unique cluster tokens**, each representing a distinct rhythmic pattern.

```
Measure 1 (e.g. two 8th notes) → Cluster 42
Measure 2 (empty rest)         → Cluster 7
Measure 3 (two 8th notes)      → Cluster 42   ← same pattern, reused
```

Each cluster has a token name: `[CLU_0000]` … `[CLU_2188]`  
Stored in: `scripts/cluster_to_patterns_tokens.txt`

---

## Step 2 — Vocabulary Extension

MuMu-LLaMA originally had **32,008 tokens**. We added our 2,189 cluster tokens:

```
Tokenizer vocab:    32,008  →  34,197
Embedding matrix:  (32008, 4096)  →  (34197, 4096)
New token rows:    warm-started with mean of existing embeddings
Output head:       expanded to match (34197 logits)
```

Also patched inside `MuMu-LLaMA/llama/mumu_llama.py`:
- Removed the `assert vocab_size == 32000 + 8` check (line 623)
- Cast `output.float()` before cross-entropy loss (prevents FP16 NaN)

---

## Step 3 — Dataset (`scripts/prepare_hierarchical_sft.py`)

Each training sample is a **sliding window** over a song:

```
Window N:
  Input  → audio waveform of this window + previous cluster token context
  Target → next cluster token to predict
```

Total dataset:
- **25,443** training windows
- **24,171 train** / **1,272 val** split

---

## Step 4 — Director Training (`scripts/train_hierarchical_mumu.py`)

**Base model**: MuMu-LLaMA  
- LLaMA-7B language model backbone  
- MERT audio encoder (understands music)  
- mu_mert_agg Conv1d aggregation  
- Audio projection into LLaMA embedding space  

**Training config:**

| Setting | Value |
|---------|-------|
| Precision | `float32` (full precision — no dtype mismatches) |
| Optimizer | AdamW, lr=1e-4, weight_decay=0.05 |
| Batch size | 1 |
| Epochs | 5 |
| GPU | 1× NVIDIA A40 (45GB VRAM) |
| VRAM used | ~30GB |
| Gradient clipping | max_norm=1.0 |

**What it learns:**  
Given MERT audio features + preceding cluster token context → predict next cluster token.  
Standard causal language model objective (cross-entropy loss).

**Loss interpretation:**

| Loss value | Meaning |
|-----------|---------|
| 10.4 | Pure random guessing (worst possible) |
| 6.2 | Start of training — already learning |
| 3–4 | Good — capturing rhythmic structure |
| 1–2 | Excellent — strong audio-pattern alignment |

---

## Step 5 — Output

After all 5 epochs complete:

```
/data/mg546924/llm_beatmap_generator/output/
├── checkpoint_final.pth    ← trained Director weights
└── tokenizer/              ← extended tokenizer (34197 tokens)
```

---

## Current Status

```
🟢 RUNNING — Epoch 1/5
   loss=6.23 → real values, no errors, no skips
   Speed: ~1.8 it/s (FP32 on A40)
   ETA: ~3.6 hours per epoch × 5 = ~18 hours total
```

---

## How to Watch Progress

```bash
# On HPC:
tail -f /data/mg546924/llm_beatmap_generator/logs/train_hierarchical_<JOBID>.log
```

```bash
# Check job status:
squeue -u $USER
```
