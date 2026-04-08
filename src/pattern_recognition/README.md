# Beatmap Pattern Recognition

This directory houses the algorithms and tooling used to mathematically evaluate the layout of generated sequence steps (Beatmaps). 
The goal is to determine if AI-generated beatmaps employ mathematically similar geometric patterns (trills, streams, jumps) as standard human-curated StepMania (`.ssc`) files.

## Directory Structure
- **`human/`**: Contains execution logs and statistical exports from analyzing native `.ssc` files made by humans (the benchmark).
- **`ai/`**: Reserved for statistical exports derived from Qwen, LLama, or DeepResonance generated outputs.
- **`recognize_patterns.py`**: The core execution script that parses files and computes chain codes.

---

## Technical Domain Logic

### Handling Multiplexed `.ssc` Files
As StepMania `.ssc` files package multiple difficulties (Beginner, Easy, Medium, Hard, Challenge) inside a single file, the pipeline employs a **Difficulty Preprocessor**. 
Instead of merging all charts together into a chaotic mess, the script scans the `.ssc` tags (specifically `#DIFFICULTY:` and `#NOTES:`) and mathematically segments each unique chart. Pattern profiles are calculated distinctly per difficulty bracket.

### Rotational Invariant Differential Chain Coding
To evaluate patterns mathematically, we utilize **Rotational Invariant First-Order Differential Chain Coding**.

In rhythm games, a "pattern" is not defined by *which exact arrow* you step on, but the *relative movement* between your feet. For example:
- `Left → Down → Left → Down` is physically identical mechanics to `Up → Right → Up → Right` (a standard trill motion).

If we simply measured raw arrows, our statistics would flag those as two completely different actions. 
By employing **Chain Coding**, we mask specific column identities underneath relational shifts.

1. **State Matrixing**: `Left = 0`, `Down = 1`, `Up = 2`, `Right = 3`.
2. **Delta Calculus**: We compare the next step to the prior step natively: `(Next - Current) mod 4`.
   - **`+1` (Clockwise Step)** (E.g. Left → Down)
   - **`+2` (Cross-Pad Jump)** (E.g. Left → Up)
   - **`+3` (Counter-Clockwise Step)** (E.g. Down → Left)
   - **`0` (Jack / Repeater)** (E.g. Left → Left)

### N-Gram Sequencing
Once an entire chart difficulty is transformed into a long string of these deltas (e.g., `+1, +2, +1, +2`), we extract **N-Grams** (chunks of 3 sequence transitions representing 4 contiguous physical steps) to find the most mathematically probable human dance motifs (like continuous staircases or alternating crossovers).
