# MIDI Measure & Subdivision Analyzer

This script performs precision grid-alignment analysis directly on the binary structure of standard `.mid` / `.midi` files. 

It calculates:
1. The absolute boundary of every **Measure** dynamically via native Time Signature / Downbeat detection.
2. The exact grid-quantized interval of every note occurring within that measure (e.g. `4th Note`, `8th Note`, `Triplets`, `16ths`, or `Off-Grid`).
3. The absolute timestamps (in milliseconds and exact seconds) defining when each fractional step lands.

## Prerequisites

The logic relies on `pretty_midi` to decode the binary data natively. You must install it first:
```bash
pip install pretty_midi
```

> **Note:** If you run into permission errors installing global pip packages on Mac, run the script within a localized Virtual Environment (e.g., `python3 -m venv venv && source venv/bin/activate && pip install pretty_midi`).

## How to Run

Execute the script from your terminal using standard Python:

```bash
# Run against the default test file
python src/music_analysis_midi/analyze_midi.py

# Run against a specific custom MIDI file
python src/music_analysis_midi/analyze_midi.py --midi "path/to/your/song.mid"
```

## Advanced Options

If you find that the grid categorizations are classifying too many notes as "Off-Grid", you can loosen the alignment margin using the `--tolerance` parameter (default `0.08` of a beat):

```bash
python src/music_analysis_midi/analyze_midi.py --midi "my_song.mid" --tolerance 0.15
```

## How to Evaluate New Music Files

Currently, this folder points purely to **MIDI binary datasets**. StepMania (`.ssc`) or raw audio (`.ogg`) files are explicitly unsupported by this module directly. 

To run analysis on your `.ogg` tracks:
1. Generate a corresponding `.mid` file utilizing frameworks like **Spotify's Basic Pitch** or direct piano roll transcription.
2. Place the resulting `.mid` file either directly into this folder or inside `src/sheet_music/`.
3. Point the script directly to its location using the `--midi` flag!
