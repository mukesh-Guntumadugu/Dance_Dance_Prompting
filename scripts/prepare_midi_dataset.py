"""
Prepare MIDI Dataset for Onset Evaluation.

For each MIDI file in a source directory, this script:
  1. Parses note and drum onsets from the MIDI file using pretty_midi.
  2. Saves those ground-truth onsets to a CSV file:
       midi_[song_name]_mp3_onsets.csv
  3. Synthesizes the MIDI into a .wav file using FluidSynth (if available),
     then converts to .mp3 using ffmpeg. Falls back to a pure-Python beep
     track using soundfile if neither tool is available.

Usage:
    python3 scripts/prepare_midi_dataset.py \\
        --midi-dir  /path/to/midi/files \\
        --out-dir   /path/to/output \\
        --soundfont /path/to/soundfont.sf2   # optional - needed for FluidSynth

Output (per song):
    <out-dir>/audio/midi_<song_name>.mp3
    <out-dir>/onsets/midi_<song_name>_mp3_onsets.csv
"""

import os
import sys
import argparse
import subprocess
import csv
import glob
import shutil

try:
    import pretty_midi
except ImportError:
    print("ERROR: pretty_midi not installed. Run:")
    print("  pip install pretty_midi")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_name(s: str) -> str:
    """Slugify a song name for use in filenames."""
    return s.replace(" ", "_").replace("'", "").replace("/", "-").replace("\\", "-")


def extract_midi_onsets(midi_path: str) -> list[float]:
    """
    Return sorted list of onset timestamps in milliseconds from a MIDI file.
    Includes: all instrument note onsets + drum hit onsets.
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    onsets_sec = set()

    for instrument in pm.instruments:
        for note in instrument.notes:
            onsets_sec.add(round(note.start, 4))

    return sorted(o * 1000.0 for o in onsets_sec)


def save_onset_csv(onsets_ms: list[float], out_path: str, song_name: str):
    """Save onsets in the required CSV format."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "onset_ms", "song"])
        for i, ms in enumerate(onsets_ms):
            writer.writerow([i, ms, song_name])
    print(f"    ✅ Saved {len(onsets_ms)} onsets → {os.path.basename(out_path)}")


def synthesize_midi(midi_path: str, out_wav: str, soundfont: str = None) -> bool:
    """
    Attempt to synthesize MIDI to WAV using FluidSynth.
    Returns True on success, False if FluidSynth is unavailable.
    """
    if shutil.which("fluidsynth") is None:
        return False
    if not soundfont or not os.path.isfile(soundfont):
        # Try to find a default soundfont
        default_paths = [
            "/usr/share/sounds/sf2/FluidR3_GM.sf2",
            "/usr/share/soundfonts/default.sf2",
        ]
        soundfont = next((p for p in default_paths if os.path.isfile(p)), None)
    if not soundfont:
        print("    ⚠️  FluidSynth found but no soundfont available — skipping synthesis.")
        return False

    cmd = [
        "fluidsynth", "-ni",
        soundfont, midi_path,
        "-F", out_wav, "-r", "44100"
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def convert_wav_to_mp3(wav_path: str, mp3_path: str) -> bool:
    """Convert WAV to MP3 using ffmpeg, if available."""
    if shutil.which("ffmpeg") is None:
        return False
    cmd = ["ffmpeg", "-y", "-i", wav_path, "-codec:a", "libmp3lame", "-qscale:a", "2", mp3_path]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def make_silent_mp3(out_mp3: str, duration_sec: float = 60.0):
    """
    Fallback: create a short silent .wav then rename to .mp3.
    FluidSynth and ffmpeg are both unavailable.
    This lets the pipeline continue even if synthesis is skipped.
    """
    try:
        import numpy as np
        import soundfile as sf
        sr = 22050
        silence = np.zeros(int(duration_sec * sr), dtype="float32")
        tmp_wav = out_mp3.replace(".mp3", "_silence.wav")
        sf.write(tmp_wav, silence, sr)
        os.rename(tmp_wav, out_mp3)
        print(f"    ⚠️  No synthesizer available — saved silent placeholder audio.")
        return True
    except Exception as e:
        print(f"    ❌  Could not create placeholder audio: {e}")
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare MIDI dataset for onset evaluation")
    parser.add_argument("--midi-dir",  required=True,  help="Directory containing .mid/.midi files")
    parser.add_argument("--out-dir",   required=True,  help="Output directory for CSVs and MP3s")
    parser.add_argument("--soundfont", default=None,   help="Optional path to .sf2 soundfont for FluidSynth")
    parser.add_argument("--no-audio",  action="store_true", help="Skip audio synthesis, only extract onsets")
    args = parser.parse_args()

    midi_files = (
        glob.glob(os.path.join(args.midi_dir, "*.mid")) +
        glob.glob(os.path.join(args.midi_dir, "*.midi"))
    )

    if not midi_files:
        print(f"❌ No MIDI files found in: {args.midi_dir}")
        sys.exit(1)

    print(f"\nFound {len(midi_files)} MIDI files in: {args.midi_dir}\n")

    onset_dir = os.path.join(args.out_dir, "onsets")
    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(onset_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    for midi_path in sorted(midi_files):
        base     = os.path.splitext(os.path.basename(midi_path))[0]
        slug     = safe_name(base)
        csv_name = f"midi_{slug}_mp3_onsets.csv"
        csv_path = os.path.join(onset_dir, csv_name)
        mp3_path = os.path.join(audio_dir, f"midi_{slug}.mp3")

        print(f"  🎵 {base}")

        # ── Step 1: Extract onsets ───────────────────────────────────────────
        try:
            onsets_ms = extract_midi_onsets(midi_path)
            save_onset_csv(onsets_ms, csv_path, base)
        except Exception as e:
            print(f"    ❌ Failed to parse MIDI: {e}")
            continue

        if args.no_audio:
            continue

        # ── Step 2: Synthesize MIDI → WAV → MP3 ─────────────────────────────
        if os.path.isfile(mp3_path):
            print(f"    ⏩ MP3 already exists — skipping synthesis.")
            continue

        tmp_wav = mp3_path.replace(".mp3", "_tmp.wav")
        synthesized = synthesize_midi(midi_path, tmp_wav, args.soundfont)

        if synthesized:
            converted = convert_wav_to_mp3(tmp_wav, mp3_path)
            if os.path.isfile(tmp_wav):
                os.remove(tmp_wav)
            if converted:
                print(f"    ✅ MP3 saved → {os.path.basename(mp3_path)}")
            else:
                # Keep the WAV as fallback
                os.rename(tmp_wav, mp3_path.replace(".mp3", ".wav"))
                print(f"    ⚠️  ffmpeg not available — saved as .wav instead.")
        else:
            # No FluidSynth: create a silent placeholder so the pipeline can still run
            pm = pretty_midi.PrettyMIDI(midi_path)
            make_silent_mp3(mp3_path, duration_sec=pm.get_end_time() + 1.0)

        print()

    print("✅ Dataset preparation complete!")
    print(f"   Onset CSVs  → {onset_dir}")
    print(f"   Audio files → {audio_dir}")


if __name__ == "__main__":
    main()
