#!/usr/bin/env python3
"""
Springtime → 4/4 Sheet Music PDF with Millisecond Timestamps

Takes a MIDI file (from basic-pitch) and renders it as a multi-page
sheet music PDF with:
  - Treble + bass clef staves
  - Notes placed at correct staff positions
  - 4/4 time signature and measure bar lines
  - Millisecond timestamps (mm:ss.mmm) on each note
"""

import os
import math
from datetime import datetime
import pretty_midi
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Ellipse, FancyBboxPatch
from matplotlib.lines import Line2D

# ─── Configuration ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIDI_FILE = os.path.join(SCRIPT_DIR, "Kommisar - Springtime_basic_pitch.mid")
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PDF = os.path.join(SCRIPT_DIR, f"Springtime_sheet_music_{TIMESTAMP}.pdf")

BPM = 120            # from MIDI metadata
TIME_SIG = (4, 4)    # 4/4 time
BEAT_DUR = 60.0 / BPM  # seconds per beat
MEASURE_DUR = BEAT_DUR * TIME_SIG[0]  # seconds per measure

# Layout constants
MEASURES_PER_LINE = 4
LINES_PER_PAGE = 4
MEASURES_PER_PAGE = MEASURES_PER_LINE * LINES_PER_PAGE

# Staff rendering
STAFF_LINE_SPACING = 1.0   # vertical distance between staff lines
NOTE_HEAD_WIDTH = 0.7
NOTE_HEAD_HEIGHT = 0.7

# Page dimensions (inches)
PAGE_W = 17
PAGE_H = 11

# ─── MIDI note → staff position mapping ─────────────────────────────
# Standard: Middle C (MIDI 60) = C4, sits on first ledger line below treble staff
# We map MIDI pitch → a "staff position" (integer or half-integer)
# where 0 = middle C, positive = up, negative = down.
# Each staff position increment = one diatonic step (line or space).

# Note names for annotation
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def midi_to_note_name(midi_num):
    """Convert MIDI number to note name + octave."""
    octave = (midi_num // 12) - 1
    note = NOTE_NAMES[midi_num % 12]
    return f"{note}{octave}"

def midi_to_staff_pos(midi_num):
    """
    Map MIDI pitch to staff position (diatonic steps from middle C).
    C4=0, D4=1, E4=2, F4=3, G4=4, A4=5, B4=6, C5=7, ...
    We map chromatically: sharps/flats share the position of their natural.
    """
    # Chromatic-to-diatonic offset within an octave
    # C  C# D  D# E  F  F# G  G# A  A# B
    chrom_to_diat = [0, 0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6]
    
    octave = (midi_num // 12) - 1
    chrom = midi_num % 12
    diat = chrom_to_diat[chrom]
    # Middle C (MIDI 60) = octave 4, diatonic 0
    return (octave - 4) * 7 + diat

def format_timestamp(seconds):
    """Format seconds as mm:ss.mmm"""
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins:02d}:{secs:06.3f}"

# ─── Drawing helpers ─────────────────────────────────────────────────

def draw_staff(ax, x_start, x_end, y_center, clef='treble'):
    """Draw 5 staff lines centered around y_center."""
    # Staff lines are at positions -2, -1, 0, +1, +2 relative to center
    for i in range(-2, 3):
        y = y_center + i * STAFF_LINE_SPACING
        ax.plot([x_start, x_end], [y, y], color='black', linewidth=0.5,
                zorder=1)

def draw_clef(ax, x, y_center, clef='treble'):
    """Draw clef indicator as styled text."""
    if clef == 'treble':
        ax.text(x, y_center, 'G', fontsize=20, ha='center', va='center',
                fontfamily='serif', fontweight='bold', fontstyle='italic',
                color='#2c3e50', zorder=5)
    else:
        ax.text(x, y_center, 'F', fontsize=20, ha='center', va='center',
                fontfamily='serif', fontweight='bold', fontstyle='italic',
                color='#2c3e50', zorder=5)

def draw_time_sig(ax, x, y_center):
    """Draw 4/4 time signature."""
    ax.text(x, y_center + STAFF_LINE_SPACING, '4', fontsize=14,
            ha='center', va='center', fontweight='bold', fontfamily='serif', zorder=5)
    ax.text(x, y_center - STAFF_LINE_SPACING, '4', fontsize=14,
            ha='center', va='center', fontweight='bold', fontfamily='serif', zorder=5)

def draw_barline(ax, x, y_treble_center, y_bass_center):
    """Draw a barline spanning both staves."""
    y_top = y_treble_center + 2 * STAFF_LINE_SPACING
    y_bot = y_bass_center - 2 * STAFF_LINE_SPACING
    ax.plot([x, x], [y_bot, y_top], color='black', linewidth=1.0, zorder=2)

def draw_note(ax, x, y, note_name, timestamp, is_sharp=False, ledger_lines=None,
              y_treble_center=None, y_bass_center=None):
    """Draw a filled note head with note name and timestamp."""
    # Note head (filled ellipse)
    ellipse = Ellipse((x, y), NOTE_HEAD_WIDTH, NOTE_HEAD_HEIGHT * 0.6,
                       angle=-15, facecolor='black', edgecolor='black',
                       linewidth=0.5, zorder=4)
    ax.add_patch(ellipse)
    
    # Note name label (above the note)
    ax.text(x, y + 0.8, note_name, fontsize=5, ha='center', va='bottom',
            color='#1a5276', fontweight='bold', zorder=5)
    
    # Timestamp label (below the note)
    ax.text(x, y - 0.9, timestamp, fontsize=4, ha='center', va='top',
            color='#922b21', fontfamily='monospace', zorder=5, rotation=45)
    
    # Draw ledger lines if needed
    if ledger_lines:
        for ly in ledger_lines:
            ax.plot([x - 0.5, x + 0.5], [ly, ly], color='black',
                    linewidth=0.5, zorder=3)

def get_ledger_lines(staff_pos, y_treble_center, y_bass_center):
    """
    Compute ledger line Y positions for notes outside the staff.
    
    Treble staff lines span positions 0 to 8 (E4=2 bottom line to F5=10 top line)
    Actually in standard notation:
      Treble bottom line = E4 (staff_pos 2), top line = F5 (staff_pos 10)
      Bass bottom line = G2 (staff_pos -12), top line = A3 (staff_pos -2)
    
    Middle C (staff_pos 0) needs one ledger line between the staves.
    """
    ledger_ys = []
    
    # Treble clef: lines at staff_pos 2, 4, 6, 8, 10 (E4, G4, B4, D5, F5)
    treble_bottom_pos = 2   # E4
    treble_top_pos = 10     # F5
    
    # Bass clef: lines at staff_pos -12, -10, -8, -6, -4 (G2, B2, D3, F3, A3)
    bass_bottom_pos = -12   # G2
    bass_top_pos = -4       # A3
    
    if staff_pos >= treble_bottom_pos - 2 or staff_pos <= bass_top_pos + 2:
        # Near or in treble clef range
        if staff_pos < treble_bottom_pos:
            # Below treble staff
            pos = treble_bottom_pos - 2
            while pos >= staff_pos:
                y = y_treble_center - 2 * STAFF_LINE_SPACING + \
                    (pos - treble_bottom_pos) * (STAFF_LINE_SPACING / 2)
                ledger_ys.append(y_treble_center + (pos - 6) * (STAFF_LINE_SPACING / 2))
                pos -= 2
        elif staff_pos > treble_top_pos:
            # Above treble staff
            pos = treble_top_pos + 2
            while pos <= staff_pos:
                ledger_ys.append(y_treble_center + (pos - 6) * (STAFF_LINE_SPACING / 2))
                pos += 2
    
    return ledger_ys


def note_to_y(staff_pos, y_treble_center, y_bass_center):
    """
    Convert staff position to Y coordinate.
    Treble staff center corresponds to B4 (staff_pos 6).
    Bass staff center corresponds to D3 (staff_pos -8).
    Each staff position = half a staff-line spacing.
    """
    # Decide which staff: if staff_pos >= -1 use treble, else bass
    if staff_pos >= -1:
        # Treble: center = staff_pos 6 (B4)
        return y_treble_center + (staff_pos - 6) * (STAFF_LINE_SPACING / 2)
    else:
        # Bass: center = staff_pos -8 (D3)
        return y_bass_center + (staff_pos - (-8)) * (STAFF_LINE_SPACING / 2)

# ─── Main rendering ─────────────────────────────────────────────────

def main():
    print(f"Loading MIDI: {MIDI_FILE}")
    midi = pretty_midi.PrettyMIDI(MIDI_FILE)
    
    # Collect all notes
    all_notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            all_notes.append({
                'pitch': note.pitch,
                'start': float(note.start),
                'end': float(note.end),
                'velocity': note.velocity,
                'name': midi_to_note_name(note.pitch),
                'staff_pos': midi_to_staff_pos(note.pitch),
                'is_sharp': '#' in midi_to_note_name(note.pitch),
            })
    
    all_notes.sort(key=lambda n: (n['start'], n['pitch']))
    print(f"Total notes: {len(all_notes)}")
    
    total_time = midi.get_end_time()
    total_measures = math.ceil(total_time / MEASURE_DUR)
    total_pages = math.ceil(total_measures / MEASURES_PER_PAGE)
    print(f"Total time: {total_time:.3f}s, Measures: {total_measures}, Pages: {total_pages}")
    
    # Group notes by measure
    notes_by_measure = {}
    for note in all_notes:
        measure_idx = int(note['start'] // MEASURE_DUR)
        if measure_idx not in notes_by_measure:
            notes_by_measure[measure_idx] = []
        notes_by_measure[measure_idx].append(note)
    
    # ─── Render PDF ──────────────────────────────────────────────
    print(f"Rendering PDF to: {OUTPUT_PDF}")
    
    with PdfPages(OUTPUT_PDF) as pdf:
        for page_num in range(total_pages):
            fig, ax = plt.subplots(1, 1, figsize=(PAGE_W, PAGE_H))
            ax.set_xlim(-1, PAGE_W * 6)
            ax.set_ylim(-PAGE_H * 7, 5)
            ax.set_aspect('equal')
            ax.axis('off')
            
            # Title on first page
            if page_num == 0:
                ax.text(PAGE_W * 3, 3, "Springtime", fontsize=24,
                        ha='center', va='center', fontweight='bold',
                        fontfamily='serif')
                ax.text(PAGE_W * 3, 1.5, "Kommisar  •  4/4 Time  •  120 BPM",
                        fontsize=12, ha='center', va='center',
                        fontfamily='serif', color='gray')
            
            for line_idx in range(LINES_PER_PAGE):
                global_line_start = page_num * MEASURES_PER_PAGE + line_idx * MEASURES_PER_LINE
                
                if global_line_start >= total_measures:
                    break
                
                # Y positions for this line's staves
                line_y_offset = -line_idx * 18 - (6 if page_num == 0 else 2)
                y_treble = line_y_offset
                y_bass = line_y_offset - 8
                
                # X positions
                x_left = 2
                x_right = PAGE_W * 6 - 2
                usable_width = x_right - x_left
                clef_width = 5
                measure_width = (usable_width - clef_width) / MEASURES_PER_LINE
                
                # Draw staves
                draw_staff(ax, x_left, x_right, y_treble, 'treble')
                draw_staff(ax, x_left, x_right, y_bass, 'bass')
                
                # Draw clefs
                draw_clef(ax, x_left + 1.5, y_treble, 'treble')
                draw_clef(ax, x_left + 1.5, y_bass, 'bass')
                
                # Draw time signature on first line
                if page_num == 0 and line_idx == 0:
                    draw_time_sig(ax, x_left + 3.5, y_treble)
                    draw_time_sig(ax, x_left + 3.5, y_bass)
                
                # Draw vertical brace at left
                y_top = y_treble + 2 * STAFF_LINE_SPACING
                y_bot = y_bass - 2 * STAFF_LINE_SPACING
                ax.plot([x_left, x_left], [y_bot, y_top], color='black',
                        linewidth=2, zorder=2)
                
                # Draw measures
                for m in range(MEASURES_PER_LINE):
                    measure_idx = global_line_start + m
                    if measure_idx >= total_measures:
                        break
                    
                    m_x_start = x_left + clef_width + m * measure_width
                    m_x_end = m_x_start + measure_width
                    
                    # Bar line at end of measure
                    draw_barline(ax, m_x_end, y_treble, y_bass)
                    
                    # Also bar line at start of first measure
                    if m == 0:
                        draw_barline(ax, m_x_start, y_treble, y_bass)
                    
                    # Measure number
                    ax.text(m_x_start + 0.5, y_treble + 3 * STAFF_LINE_SPACING,
                            str(measure_idx + 1), fontsize=7, ha='center',
                            va='bottom', color='gray')
                    
                    # Measure start time
                    measure_start_time = measure_idx * MEASURE_DUR
                    ax.text(m_x_start + 0.5,
                            y_bass - 3 * STAFF_LINE_SPACING,
                            format_timestamp(measure_start_time),
                            fontsize=5, ha='center', va='top',
                            color='#7d3c98', fontfamily='monospace')
                    
                    # Draw notes in this measure
                    notes = notes_by_measure.get(measure_idx, [])
                    if not notes:
                        continue
                    
                    for note in notes:
                        # X position within measure (based on beat position)
                        beat_in_measure = (note['start'] - measure_start_time) / BEAT_DUR
                        note_x = m_x_start + 1.5 + (beat_in_measure / TIME_SIG[0]) * (measure_width - 3)
                        
                        # Y position based on staff position
                        note_y = note_to_y(note['staff_pos'], y_treble, y_bass)
                        
                        # Ledger lines
                        ledger = get_ledger_lines(note['staff_pos'], y_treble, y_bass)
                        
                        # Draw the note
                        draw_note(ax, note_x, note_y, note['name'],
                                  format_timestamp(note['start']),
                                  is_sharp=note['is_sharp'],
                                  ledger_lines=ledger,
                                  y_treble_center=y_treble,
                                  y_bass_center=y_bass)
            
            # Page number
            ax.text(PAGE_W * 3, -PAGE_H * 7 + 1,
                    f"Page {page_num + 1} of {total_pages}", fontsize=8,
                    ha='center', va='bottom', color='gray')
            
            plt.tight_layout(pad=0.5)
            pdf.savefig(fig, dpi=150)
            plt.close(fig)
            print(f"  Page {page_num + 1}/{total_pages} rendered")
    
    file_size = os.path.getsize(OUTPUT_PDF)
    print(f"\n✅ Sheet music saved to: {OUTPUT_PDF}")
    print(f"   File size: {file_size / 1024:.1f} KB")
    print(f"   Pages: {total_pages}")

if __name__ == '__main__':
    main()
