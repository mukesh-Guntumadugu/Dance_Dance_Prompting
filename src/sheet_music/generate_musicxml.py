#!/usr/bin/env python3
"""
Springtime MIDI → MusicXML Converter (Improved Version)

Converts the improved basic-pitch MIDI transcription into MusicXML format.
Uses the high-sensitivity transcription for better accuracy.

Timestamps are embedded as lyrics on each note.
"""

import os
from datetime import datetime
import music21
from music21 import converter, meter, tempo, metadata, instrument

# ─── Configuration ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_FILE = os.path.join(SCRIPT_DIR, "improved",
                         "Kommisar - Springtime_basic_pitch.mid")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_XML = os.path.join(SCRIPT_DIR, f"Springtime_sheet_music_{TIMESTAMP}.musicxml")

BPM = 120


def format_timestamp(seconds):
    """Format seconds as mm:ss.mmm"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"


def main():
    print(f"Loading MIDI: {MIDI_FILE}")

    # Parse the MIDI file with music21
    score = converter.parse(MIDI_FILE)
    print(f"Parsed score with {len(score.parts)} part(s)")

    # Set metadata
    score.metadata = metadata.Metadata()
    score.metadata.title = "Springtime"
    score.metadata.composer = "Kommisar"

    # Process each part
    total_notes = 0
    for part_idx, part in enumerate(score.parts):
        # Remove existing time signatures
        for ts in part.recurse().getElementsByClass(meter.TimeSignature):
            part.remove(ts, recurse=True)

        # Insert 4/4 at the beginning
        part.insert(0, meter.TimeSignature('4/4'))
        part.insert(0, tempo.MetronomeMark(number=BPM))

        # Add timestamps as lyrics
        beat_duration = 60.0 / BPM
        note_count = 0

        for element in part.recurse().notesAndRests:
            if isinstance(element, (music21.note.Note, music21.chord.Chord)):
                offset_beats = float(element.getOffsetInHierarchy(part))
                timestamp_sec = offset_beats * beat_duration
                ts_text = format_timestamp(timestamp_sec)
                element.addLyric(ts_text)
                note_count += 1

        total_notes += note_count
        print(f"  Part {part_idx + 1}: {note_count} notes with timestamps")

    # Make measures
    for part in score.parts:
        if not part.hasMeasures():
            part.makeMeasures(inPlace=True)

    # Export to MusicXML
    print(f"\nExporting MusicXML to: {OUTPUT_XML}")
    score.write('musicxml', fp=OUTPUT_XML)
    file_size = os.path.getsize(OUTPUT_XML)
    print(f"  File size: {file_size / 1024:.1f} KB")

    print(f"\n✅ Done!")
    print(f"   MusicXML: {OUTPUT_XML}")
    print(f"   Total notes with timestamps: {total_notes}")
    print(f"\n💡 Open in MuseScore, Finale, Sibelius, or noteflight.com to view")


if __name__ == '__main__':
    main()
