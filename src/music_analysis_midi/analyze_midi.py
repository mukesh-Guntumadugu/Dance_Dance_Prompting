#!/usr/bin/env python3
import os
import sys
import argparse
import numpy as np

try:
    import pretty_midi
except ImportError:
    print(" Error: 'pretty_midi' is required. Run: pip install pretty_midi")
    sys.exit(1)

def build_grid_lookups():
    """Build a mapping of valid 4/4 grid locations to their note type."""
    # A single beat ranges from [0.0 to 1.0)
    # We span 4 beats (0.0 to < 4.0)
    grid = []
    
    for b in range(4):
        # Quarter Notes (4ths)
        grid.append({"offset": float(b), "label": "4th (Quarter)", "priority": 1})
        
        # 8th Notes
        grid.append({"offset": b + 0.5, "label": "8th Note", "priority": 2})
        
        # 16th Notes
        grid.append({"offset": b + 0.25, "label": "16th Note", "priority": 3})
        grid.append({"offset": b + 0.75, "label": "16th Note", "priority": 3})
        
        # Triplets (12ths of a whole measure, so 1/3 and 2/3 of a beat)
        grid.append({"offset": b + 0.333333, "label": "Triplet (1/3)", "priority": 4})
        grid.append({"offset": b + 0.666667, "label": "Triplet (2/3)", "priority": 4})

    # Sort so closest match logic cleanly maps
    return sorted(grid, key=lambda x: x["offset"])


def analyze_midi_measures(midi_path, tolerance=0.08):
    """
    Parses a MIDI binary to extract precise Measure timings and Note subdivisions.
    """
    if not os.path.exists(midi_path):
        print(f" File not found: {midi_path}")
        return

    print("="*60)
    print(f"🎵 MIDI Measure Analysis: {os.path.basename(midi_path)}")
    print("="*60)

    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception as e:
        print(f" Failed to parse MIDI: {e}")
        return

    # Extract global properties
    tempo_changes = pm.get_tempo_changes()
    tempos = tempo_changes[1]
    avg_tempo = np.mean(tempos) if len(tempos) > 0 else 120.0
    
    # 1. Total Measures (Using downbeats)
    downbeats = pm.get_downbeats()
    beats = pm.get_beats()
    
    if len(downbeats) == 0:
        print("  No native Time Signature found preventing exact downbeat alignment.")
        print("  Falling back to estimated 4/4 beats...")
        # Artificially create downbeats every 4 beats
        if len(beats) > 0:
            downbeats = beats[::4]
        else:
            print(" No beats detected. Cannot analyze grid.")
            return

    # Treat the end of the song as the absolute last marker
    end_time = pm.get_end_time()
    downbeats = list(downbeats)
    if downbeats[-1] < end_time:
        downbeats.append(end_time)

    total_measures = len(downbeats) - 1
    print(f"  Duration:           {end_time:.2f} seconds")
    print(f" Estimated BPM:      {avg_tempo:.1f}")
    print(f" Total Measures:     {total_measures}")
    print("-"*60)

    # Gather all note onsets across all tracks
    all_onsets_sec = set()
    for inst in pm.instruments:
        for note in inst.notes:
            all_onsets_sec.add(note.start)
    all_onsets_sec = sorted(list(all_onsets_sec))

    if not all_onsets_sec:
        print(" No notes found in this MIDI.")
        return

    grid_targets = build_grid_lookups()

    summary_stats = {
        "4th (Quarter)": 0,
        "8th Note": 0,
        "16th Note": 0,
        "Triplet (1/3)": 0,
        "Triplet (2/3)": 0,
        "Off-Grid (Unquantized)": 0
    }

    print("\n Measure-by-Measure Breakdown Trace:")
    
    # Full pass for Global Counts
    all_measure_breakdowns = []
    
    for m in range(total_measures):
        m_start = downbeats[m]
        m_end = downbeats[m+1]
        m_dur = m_end - m_start
        if m_dur <= 0: continue
        q_dur = m_dur / 4.0 
        
        notes = [t for t in all_onsets_sec if m_start <= t < m_end]
        
        measure_stats = {
            "4th (Quarter)": 0,
            "8th Note": 0,
            "16th Note": 0,
            "Triplet (1/3)": 0,
            "Triplet (2/3)": 0,
            "Off-Grid (Unquantized)": 0
        }
        
        for t in notes:
            offset_beats = (t - m_start) / q_dur
            best_match = None
            min_dist = float('inf')
            for target in grid_targets:
                dist = abs(offset_beats - target["offset"])
                if dist < min_dist:
                    min_dist = dist
                    best_match = target
                elif dist == min_dist and best_match and target["priority"] < best_match["priority"]:
                    best_match = target
            
            if min_dist <= tolerance:
                final_label = best_match["label"]
            else:
                final_label = "Off-Grid (Unquantized)"
                
            summary_stats[final_label] += 1
            measure_stats[final_label] += 1
            
            all_measure_breakdowns.append({
                "measure": m+1,
                "timestamp_sec": t,
                "timestamp_ms": round(t * 1000),
                "label": final_label
            })
            
        # Format the specific output for THIS measure
        active_parts = []
        for label, count in measure_stats.items():
            if count > 0:
                active_parts.append(f"{count}x {label}")
                
        if len(active_parts) > 0:
            print(f"  [Measure {m+1:>3}] → " + ", ".join(active_parts))
        else:
            print(f"  [Measure {m+1:>3}] → (Empty) 0 notes")

    print("\n" + "="*60)
    print(" FINAL SUBDIVISION COUNTS ACROSS ALL MEASURES:")
    for label, amt in summary_stats.items():
        if amt > 0:
            print(f"  - {label:<22} : {amt} notes")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", type=str, default="src/sheet_music/Kommisar - Springtime_basic_pitch.mid",
                        help="Path to the .mid/.midi file to analyze")
    parser.add_argument("--tolerance", type=float, default=0.08,
                        help="Detection tolerance in beats (default 0.08)")
    
    args = parser.parse_args()
    
    analyze_midi_measures(args.midi, args.tolerance)
