
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

def plot_sheet_music(audio_path, original_notes, generated_notes, original_label="Original", generated_label="Generated", output_file="beatmap_sheet_music.png", start_time=20, duration=10):
    """
    Plot waveform and notes with specific labels.
    """
    print(f"Loading audio {audio_path}...")
    y, sr = librosa.load(audio_path, sr=None, offset=start_time, duration=duration)
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # 1. Waveform
    audio_filename = os.path.basename(audio_path)
    librosa.display.waveshow(y, sr=sr, ax=ax1, alpha=0.6, color='gray')
    ax1.set_title(f"Audio Track: {audio_filename}\nWindow: {start_time}s - {start_time+duration}s", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Amplitude")
    
    # Overlay Onsets
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    # Adjust for plot which starts at 0 (relative to loading offset)
    # Actually waveshow handles 'x_axis' if we provide it, but default is indices.
    # Let's simple mapping: x axis is time relative to start_time.
    
    ax1.vlines(onset_times, -1, 1, color='r', linestyle='--', alpha=0.5, label='Onsets')
    ax1.legend(loc='upper right')
    
    # Helper to plot notes
    arrow_map = {0: '$\u2190$', 1: '$\u2193$', 2: '$\u2191$', 3: '$\u2192$'} # Left, Down, Up, Right
    col_colors = {0: '#E91E63', 1: '#00BCD4', 2: '#4CAF50', 3: '#FFC107'} # Pink, Cyan, Green, Yellow
    
    def plot_track(ax, notes, title):
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylim(-0.5, 3.5)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Left', 'Down', 'Up', 'Right'])
        ax.grid(True, axis='y', linestyle='-', alpha=0.3)
        ax.grid(True, axis='x', linestyle='--', alpha=0.3)
        
        # Filter notes in time window
        visible_notes = [n for n in notes if start_time <= n[0] < start_time + duration]
        
        for t, col, char in visible_notes:
            rel_time = t - start_time
            # Using simple markers for now. 
            # Marker styles: < v ^ >
            marker = 'o'
            if col == 0: marker = '<'
            elif col == 1: marker = 'v'
            elif col == 2: marker = '^'
            elif col == 3: marker = '>'
            
            color = col_colors.get(col, 'black')
            if char == 'M': 
                marker = 'X'
                color = 'red'
            elif char == '2': # Hold head
                marker = 's' # square
            
            ax.plot(rel_time, col, marker=marker, markersize=15, color=color, markeredgecolor='black')
    
    # 2. Original Beatmap
    plot_track(ax2, original_notes, f"Original Beatmap (Ground Truth)\nFile: {original_label}")
    
    # 3. Generated Beatmap
    plot_track(ax3, generated_notes, f"Generated Beatmap (AI)\nFile: {generated_label}")
    
    # X Axis
    # Convert samples to time for ax1? Librosa waveshow does it automatically?
    # Librosa waveshow plots time on x-axis relative to 0 of the y array.
    # So all axes share x-axis 0..duration.
    ax3.set_xlabel("Time (seconds relative to window start)", fontsize=12)
    
    # Add Date/Time Footer
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plt.figtext(0.99, 0.01, f"Generated: {timestamp}", ha="right", fontsize=10, fontstyle='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Sheet music saved to {output_file}")


    
def generate_filename_with_timestamp(original_filename):
    """
    Inserts a current timestamp before the file extension.
    
    Example: 'report.csv' -> 'report_2026-02-14_14-50-00.csv'
    """
    # Get current date and time
    now = datetime.datetime.now()
    # Format the date and time as a string suitable for filenames
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Separate the base name and extension
    base, extension = os.path.splitext(original_filename)
    
    # Construct the new filename
    new_filename = f"{base}_{timestamp}{extension}"
    
    return new_filename



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
    
    # Plot a specific interesting section
    # Let's find a dense section or just the middle.
    # Start around 48s (where sample start was in .ssc)
    START_TIME = 48.0
    DURATION = 15.0 
    
    
    # Generate timestamped filename
    output_filename = "beatmap_sheet_music.png"
    output_filename = generate_filename_with_timestamp(output_filename)
    
    plot_sheet_music(AUDIO_FILE, orig_notes, gen_notes, 
                    original_label=ORIG_FILE, 
                    generated_label=GEN_FILE,
                    output_file=output_filename,
                    start_time=START_TIME, duration=DURATION)

if __name__ == "__main__":
    main()
