
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import os
import datetime

def load_beatmap_measures(filepath):
    """Load beatmap measures from file."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    measures = []
    current_measure = []
    
    for line in content.strip().split('\n'):
        line = line.strip()
        if not line: continue
        if line == ',':
            if current_measure:
                measures.append(current_measure)
                current_measure = []
        elif len(line) == 4 and all(c in '01234M' for c in line):
            current_measure.append(line)
            
    if current_measure:
        measures.append(current_measure)
        
    return measures

def get_note_data(measures, bpm, offset):
    """
    Convert beatmap measures to list of (timestamp, column_index, note_type)
    Column index: 0=Left, 1=Down, 2=Up, 3=Right
    """
    notes = []
    beats_per_measure = 4
    seconds_per_beat = 60.0 / bpm
    start_time = offset
    
    for measure_idx, measure in enumerate(measures):
        measure_start_beat = measure_idx * beats_per_measure
        lines_in_measure = len(measure)
        if lines_in_measure == 0: continue
        
        beats_per_line = beats_per_measure / lines_in_measure
        
        for line_idx, line in enumerate(measure):
            beat_time = measure_start_beat + (line_idx * beats_per_line)
            timestamp = start_time + (beat_time * seconds_per_beat)
            
            for col_idx, char in enumerate(line):
                if char in '1234M':
                    notes.append((timestamp, col_idx, char))
                    
    return notes

def plot_music_and_beatmap(audio_path, original_notes, generated_notes, original_label="Original", generated_label="Generated", output_file="music_vs_beatmap.png", start_time=48, duration=10):
    """
    Plot Chromagram (Musical Notes) and Beatmap (Rhythm).
    """
    print(f"Loading audio {audio_path}...")
    y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=duration)
    
    # Calculate Chroma (Pitch Classes)
    # This shows the energy in each of the 12 chromatic pitches (C, C#, D, etc.)
    print("Extracting musical pitch content (Chromagram)...")
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # 1. Chromagram (The "Sheet Music")
    audio_filename = os.path.basename(audio_path)
    img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax1, cmap='coolwarm')
    ax1.set_title(f"Musical Content (Pitches/Chords)\nFile: {audio_filename}\nWindow: {start_time}s - {start_time+duration}s", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time (s) relative to window")
    fig.colorbar(img, ax=ax1, label='Energy')
    
    # 2. Beatmaps (Rhythm)
    # We plot dots for every note event in time.
    # Top half of ax2: Original
    # Bottom half of ax2: Generated
    
    ax2.set_title("Beatmap Rhythm Alignment", fontsize=14, fontweight='bold')
    ax2.set_yticks([0.25, 0.75])
    ax2.set_yticklabels([f"Generated (AI)\n{os.path.basename(generated_label)}", f"Original\n{os.path.basename(original_label)}"])
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis='x', linestyle='--', alpha=0.5)
    
    # Helper to plot rhythm dots
    def plot_rhythm_on_timeline(notes, y_level, color, marker):
        visible_notes = [n for n in notes if start_time <= n[0] < start_time + duration]
        timestamps = [n[0] - start_time for n in visible_notes]
        # Jitter y slightly to show density if multiple notes at same time (chords/jumps)
        # Actually simplified: just one dot per timestamp? Or stack them?
        # Let's stack them slightly vertical to see chords.
        
        # Group by timestamp to see concurrency
        from collections import defaultdict
        grouped = defaultdict(list)
        for t, col, char in visible_notes:
            grouped[t].append(col)
            
        for t_abs, cols in grouped.items():
            t_rel = t_abs - start_time
            count = len(cols)
            # Plot a marker. Size/Color depends on count (Single vs Jump vs Hand)
            
            c = color
            s = 100
            if count == 2: 
                c = 'orange' # Jump
                s = 150
            elif count >= 3: 
                c = 'red' # Hand
                s = 200
                
            ax2.scatter(t_rel, y_level, color=c, s=s, marker=marker, edgecolors='black', alpha=0.9, zorder=10)
            
            # Draw vertical line connecting to music?
            # ax2.axvline(t_rel, color=c, alpha=0.3, linestyle=':')

    # Plot Original (Top)
    plot_rhythm_on_timeline(original_notes, 0.75, '#4CAF50', 'D') # Green Diamonds
    
    # Plot Generated (Bottom)
    plot_rhythm_on_timeline(generated_notes, 0.25, '#2196F3', 'o') # Blue Circles
    
    ax2.set_xlabel("Time (seconds relative to window start)", fontsize=12)
    
    # Add Legend for Rhythm
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Single Note', markerfacecolor='#4CAF50', markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Jump (2 notes)', markerfacecolor='orange', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Hand (3+ notes)', markerfacecolor='red', markersize=15),
    ]
    ax2.legend(handles=legend_elements, loc='upper right')

    # Add Date/Time Footer
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.01, f"Generated: {timestamp}", ha="right", fontsize=10, fontstyle='italic', color='gray')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Analysis saved to {output_file}")

def generate_filename_with_timestamp(original_filename):
    """
    Inserts a current timestamp before the file extension.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    base, extension = os.path.splitext(original_filename)
    return f"{base}_{timestamp}{extension}"

def main():
    # Configuration
    GEN_FILE = "generated_MechaTribe_eassy_20260211_012048.txt"
    ORIG_FILE = "MechaTribe_original_easy.ssc.text"
    AUDIO_FILE = "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ogg"
    
    # Metadata
    BPM = 180.0
    OFFSET = -0.028
    
    # Load
    print("Loading beatmaps...")
    gen_measures = load_beatmap_measures(GEN_FILE)
    orig_measures = load_beatmap_measures(ORIG_FILE)
    
    gen_notes = get_note_data(gen_measures, BPM, OFFSET)
    orig_notes = get_note_data(orig_measures, BPM, OFFSET)
    
    # Focus Window
    START_TIME = 48.0
    DURATION = 10.0
    
    output_filename = "music_vs_beatmap.png"
    output_filename = generate_filename_with_timestamp(output_filename)
    
    plot_music_and_beatmap(AUDIO_FILE, orig_notes, gen_notes, 
                          original_label=ORIG_FILE, 
                          generated_label=GEN_FILE,
                          output_file=output_filename,
                          start_time=START_TIME, duration=DURATION)

if __name__ == "__main__":
    main()
