# Qwen Audio Probing Tool (`probe_qwen.py`)

## What is this tool?
Neural Network audio embeddings are essentially unreadable arrays of floating point numbers (the LLM "Black Box"). The `probe_qwen.py` script exists to mathematically prove what features Qwen *actually* understands and cares about when listening to an audio file, specifically for rhythmic games.

This script intercepts the Whisper-based `audio_tower` inside Qwen-Audio before the LLM turns the features into text. It mathematically crushes those hidden dimensions using **PCA (Principal Component Analysis)** into a single, time-based activation line.

By graphing this AI "Activation Spike" over the exact same timeline as a rigid `librosa` mathematical extraction of the song's **Onsets** and **BPM**, you can physically see if Qwen's internal spikes natively align with the drums on a graph.

## Files
1. **`src/probe_qwen.py`**: The main PyTorch/Librosa extraction Python script.
2. **`slurm_run_probe_qwen.sh`**: The Slurm batched job array that iterates across the `Fraxtil's Arrow Arrangements` music directory running the visualizer.

## How it Works
1. **The Ground Truth Curve**: Extracts a rigid mathematical Onset/BPM curve from the `.ogg` file using Librosa.
2. **The Output Target**: Skips the text-processor and feeds the `.ogg` directly into Qwen's internal embedding structure.
3. **The Math**: We `try/except` extract the deepest `hidden_states` layers from the Neural Encoders.
4. **The Visual**: Maps both curves into high-resolution Matplotlib PNG charts so you can see the correlation.

## How to Run (On HPC Cluster)

To process the entire Fraxtil directory, simply submit the Slurm batch file from the root directory of your project:
```bash
sbatch slurm_run_probe_qwen.sh
```

**Results:**
The batch job will create a new directory named `results_qwen_probe_fraxtil/` at the root of your project. After the batch completes (roughly ~3-5 minutes per song), that folder will be populated with PNG plots that you can download directly to your local Macbook and inspect!

---

## 🛑 Model Benchmarking Conclusions (Qwen vs. Flamingo vs. MuMu vs. DeepSeek)

Through rigorous PCA extraction and API probing of mult-modal candidates for the architecture's "Director" module, we arrived at the following conclusions:

*   ### ✅ **MuMu-LLaMA (Winner)**
    *   **Why:** Natively incorporates the MERT (Music Extraction Representation Toolkit) encoder.
    *   **Result:** Completely swallows raw audio and converts it perfectly into its Unified Latent Space. Its neuronal arrays precisely aligned with Librosa tracking. It is the chosen candidate for step-chart mapping.

*   ### ❌ **Music-Flamingo**
    *   **Why not:** Relies on the AudioMAE backbone. While it yields massive temporal dimensions (3500+ tokens), its architectural integration is slightly less geared toward raw rhythmic mapping than MuMu's MERT interface.

*   ### ❌ **Qwen-Audio**
    *   **Why not:** Whisper-based audio projection. Highly aligned for transcription/speech, but severely lacks the micro-timing and latency required for raw drum detection (compresses temporal array to only 750 tokens).

*   ### ❌ **DeepSeek-V3.2 API**
    *   **Result:** Hard-crashed with `unknown variant input_audio, expected text`. The physical public DeepSeek endpoint structurally rejects soundwaves. Thus, it cannot be our standalone zero-shot Director model without building a complex text-math translator proxy first.

---

## 🛑 Troubleshooting & Terminal Management

When running long Python scripts or API calls (like `probe_deepseek_api.py`), the process may sometimes hang due to network connectivity issues or massive file sizes. Here is how to manage frozen processes in your Mac/Linux terminal:

*   **`Ctrl + C` (Cancel/Kill):** This completely halts the script immediately. If the script is frozen or hanging, press `Ctrl + C` to throw a `KeyboardInterrupt` and kill the execution safely.
*   **`Ctrl + Z` (Suspend/Pause):** This does **NOT** kill the program! It "suspends" it and throws it frozen into the background, where it will continue to eat up memory.
*   **Managing Suspended Jobs:** If you accidentally press `Ctrl + Z`:
    1. Type `jobs` in your terminal to see the frozen programs.
    2. To kill the frozen program in slot `[1]`, type `kill -9 %1`.
