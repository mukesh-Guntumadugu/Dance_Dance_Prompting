
import numpy as np
import librosa
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Add src to python path to import generation module if needed
# (metrics are self-contained here to avoid dependency issues)

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

def get_note_times(measures, bpm, offset):
    """
    Convert beatmap measures to timestamps.
    Assumes 4/4 time signature.
    """
    note_times = []
    beats_per_measure = 4
    seconds_per_beat = 60.0 / bpm
    
    current_time = -offset # Start time (offset is usually negative in .ssc meaning song starts after)
    # Wait, Offset in .ssc usually means: time in seconds where beat 0 occurs.
    # So Time = Offset + BeatIndex * SecondsPerBeat.
    # But usually Offset is negative, meaning the song starts *before* the first beat? 
    # Or positive?
    # Spec: "Offset is the time in seconds that the first beat of the song occurs."
    # If Offset is -0.028, beat 0 is at -0.028s.
    
    start_time = offset 
    
    for measure_idx, measure in enumerate(measures):
        measure_start_beat = measure_idx * beats_per_measure
        lines_in_measure = len(measure)
        if lines_in_measure == 0: continue
        
        beats_per_line = beats_per_measure / lines_in_measure
        
        for line_idx, line in enumerate(measure):
            # Check if there is a note (any non-zero)
            # Mines (M) are also notes for timing purposes? Or ignore?
            # User wants "Physical Feasibility" so M counts as object.
            # But for Rhythm, maybe just taps.
            # Let's count any tap (1, 2, 4) or hold head (2).
            # Ignore mines 'M' for audio onset matching usually, but let's include for now.
            has_note = any(c in '1234' for c in line)
            
            if has_note:
                beat_time = measure_start_beat + (line_idx * beats_per_line)
                timestamp = start_time + (beat_time * seconds_per_beat)
                note_times.append(timestamp)
                
    return np.array(note_times)

def analyze_rhythmic_density_correlation(audio_path, note_times, duration):
    """
    Correlate 1-second windowed NPS with RMS energy.
    """
    print("Loading audio for density correlation...")
    y, sr = librosa.load(audio_path, sr=None)
    
    # Calculate RMS energy
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    
    # Resample RMS to 1-second intervals for correlation
    # Or easier: Bin notes into 1s windows, average RMS in 1s windows.
    
    n_bins = int(duration) + 1
    nps_bins = np.zeros(n_bins)
    rms_bins = np.zeros(n_bins)
    
    # NPS
    for t in note_times:
        if 0 <= t < duration:
            nps_bins[int(t)] += 1
            
    # RMS
    # Map RMS frames to bins
    for i, t in enumerate(rms_times):
        if 0 <= t < duration:
            rms_bins[int(t)] += rms[i]
            
    # Normalize RMS (sum to average)
    frames_per_sec = sr / hop_length
    rms_bins /= frames_per_sec # approx average
    
    # Filter out silence/intro/outro to avoid skewing?
    # Let's keep all valid audio.
    
    if np.sum(nps_bins) == 0:
        return 0.0, nps_bins, rms_bins
        
    correlation = np.corrcoef(nps_bins, rms_bins)[0, 1]
    
    return correlation, nps_bins, rms_bins

def analyze_jitter(audio_path, note_times):
    """
    Calculate temporal offset (jitter) from nearest onsets.
    """
    print("Detecting onsets for jitter analysis...")
    y, sr = librosa.load(audio_path, sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    
    if len(onset_times) == 0 or len(note_times) == 0:
        return 0.0, 0.0
        
    # Find nearest onset for each note
    offsets = []
    
    for t in note_times:
        # Simple nearest neighbor
        idx = np.searchsorted(onset_times, t)
        
        # Check left and right
        left_diff = float('inf')
        right_diff = float('inf')
        
        if idx > 0:
            left_diff = abs(t - onset_times[idx-1])
        if idx < len(onset_times):
            right_diff = abs(t - onset_times[idx])
            
        offset = min(left_diff, right_diff)
        offsets.append(offset)
        
    return np.mean(offsets), np.std(offsets)

def analyze_patterns(measures):
    """
    Analyze jump/hand/quad distribution.
    """
    counts = {
        'single': 0,
        'jump': 0,
        'hand': 0,
        'quad': 0,
        'mine': 0,
        'roll': 0,
        'lift': 0
    }
    
    for measure in measures:
        for line in measure:
            # Count concurrent notes
            notes = sum(1 for c in line if c in '1234')
            mines = sum(1 for c in line if c == 'M')
            
            if notes == 1: counts['single'] += 1
            elif notes == 2: counts['jump'] += 1
            elif notes >= 3: counts['hand'] += 1 # 3 or 4
            # Specific quad check if needed, but 'hand' covers 3+
            if notes == 4: counts['quad'] += 1
            
            if mines > 0: counts['mine'] += 1
            
    return counts

def check_feasibility(measures):
    """
    Check for impossible moves.
    """
    impossible_counts = 0
    issues = []
    
    for m_idx, measure in enumerate(measures):
        for l_idx, line in enumerate(measure):
            notes = sum(1 for c in line if c in '1234')
            
            # Criterion 1: 3+ notes (Hands) in Easy difficulty
            # Technically possible with hands, but "impossible" for feet-only play
            if notes >= 3:
                impossible_counts += 1
                issues.append(f"Measure {m_idx+1}, Line {l_idx+1}: {notes} concurrent notes (Hand/Quad)")
                
            # Criterion 2: Left+Right+Down (Candle) - covered by >=3 check
            
            # Criterion 3: Impossible jumps? (e.g. Left+Right wide split is Jump, allowed. Left+Up is Jump. All 2-combos are technically valid Jumps.)
            
    return impossible_counts, issues

def create_visualization(nps_bins, rms_bins, offsets, patterns, output_file="advanced_score_card.png"):
    """
    Create a visual score card of the metrics.
    """
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
    
    # 1. NPS vs Energy (Top Wide)
    ax1 = plt.subplot(gs[0, :])
    times = np.arange(len(nps_bins))
    
    # Plot RMS Energy (Background)
    ax1.fill_between(times, 0, rms_bins / max(rms_bins) * max(nps_bins), color='gray', alpha=0.3, label='Audio Energy (RMS)')
    
    # Plot NPS (Foreground)
    ax1.plot(times, nps_bins, color='blue', linewidth=2, label='Note Density (NPS)')
    
    # Formatting
    ax1.set_title("Rhythmic Density vs Audio Energy", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Notes per Second / Norm. Energy")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 2. Jitter Histogram (Bottom Left)
    ax2 = plt.subplot(gs[1, 0])
    
    # Convert to ms
    offsets_ms = [o * 1000 for o in offsets]
    
    # Histogram
    n, bins, patches = ax2.hist(offsets_ms, bins=30, color='purple', alpha=0.7, edgecolor='black')
    
    # Add mean/std lines
    mean_val = np.mean(offsets_ms)
    std_val = np.std(offsets_ms)
    ax2.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.1f}ms')
    ax2.axvline(30, color='green', linestyle='dotted', linewidth=2, label='Ideal (<30ms)')
    
    ax2.set_title("Temporal Jitter Distribution", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Offset from Onset (ms)")
    ax2.set_ylabel("Count")
    ax2.legend()
    
    # 3. Pattern Distribution (Bottom Right)
    ax3 = plt.subplot(gs[1, 1])
    
    labels = ['Single', 'Jump (2)', 'Hand (3+)', 'Mine']
    values = [patterns['single'], patterns['jump'], patterns['hand'] + patterns['quad'], patterns['mine']]
    colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336'] # Green, Blue, Orange, Red
    
    bars = ax3.bar(labels, values, color=colors, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{int(height)}',
                ha='center', va='bottom')
                
    ax3.set_title("Pattern Distribution", fontsize=14, fontweight='bold')
    ax3.set_ylabel("Count")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"\nVisualization saved to {output_file}")

def main():
    # Configuration
    BEATMAP_FILE = "generated_MechaTribe_eassy_20260211_012048.txt"
    AUDIO_FILE = "src/musicForBeatmap/MechaTribe Assault/Mecha-Tribe Assault.ogg"
    
    # Metadata from .ssc (Manual or could be parsed)
    # #OFFSET:-0.028000;
    # #BPMS:0.000=180.000;
    BPM = 180.0
    OFFSET = -0.028
    
    if not os.path.exists(BEATMAP_FILE):
        print(f"Error: {BEATMAP_FILE} not found.")
        return
        
    if not os.path.exists(AUDIO_FILE):
        print(f"Error: {AUDIO_FILE} not found.")
        return
        
    print(f"Analyzing {BEATMAP_FILE}...")
    
    # 1. Load Data
    measures = load_beatmap_measures(BEATMAP_FILE)
    if not measures:
        print("Error: No measures found in beatmap.")
        return
        
    note_times = get_note_times(measures, BPM, OFFSET)
    duration = note_times[-1] if len(note_times) > 0 else 0
    
    print(f"Loaded {len(note_times)} notes. Duration approx {duration:.2f}s.")
    
    # 2. Rhythmic Density Correlation
    corr, nps, rms = analyze_rhythmic_density_correlation(AUDIO_FILE, note_times, duration)
    
    # 3. Jitter
    mean_jitter, std_jitter = analyze_jitter(AUDIO_FILE, note_times)
    # Re-calculate raw offsets for plotting (inefficient but cleaner logic separation)
    # Or modify analyze_jitter to return raw offsets. Let's modify it.
    # Actually, simpler to just re-run the offset calculation logic here or inside create_visualization 
    # if we passed note_times and audio path.
    # PRO TIP: Let's quickly patch analyze_jitter to return offsets list too.
    
    # PATCHING analyze_jitter on the fly:
    # Actually, let's just copy the offset logic for the viz since I can't easily change the return signature of the function above without editing it too.
    # Wait, I CAN edit it too. I'm an AI. I'll just redo calculation here for safety/simplicity in this specific edit block.
    
    # Quick Offset Recalculation for Plotting
    y, sr = librosa.load(AUDIO_FILE, sr=None)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    raw_offsets = []
    if len(onset_times) > 0:
        for t in note_times:
            idx = np.searchsorted(onset_times, t)
            left = abs(t - onset_times[idx-1]) if idx > 0 else float('inf')
            right = abs(t - onset_times[idx]) if idx < len(onset_times) else float('inf')
            raw_offsets.append(min(left, right))
    
    # 4. Patterns
    patterns = analyze_patterns(measures)
    
    # 5. Feasibility
    impossible, issues = check_feasibility(measures)
    
    # 6. Visualization
    create_visualization(nps, rms, raw_offsets, patterns)
    
    # --- REPORT ---
    print("\n" + "="*50)
    print("QUANTITATIVE MEASUREMENT REPORT")
    print("="*50)
    
    print("\n1. Rhythmic Density Correlation")
    print(f"   Correlation (NPS vs Audio Energy): {corr:.4f}")
    if corr > 0.5: print("   -> STRONG POSITIVE correlation. (Good)")
    elif corr > 0.2: print("   -> WEAK POSITIVE correlation. (Acceptable)")
    else: print("   -> NO/NEGATIVE correlation. (Potential Issue: Random placement?)")
    
    print("\n2. Temporal Offset (Jitter)")
    print(f"   Mean Jitter (dist to nearest onset): {mean_jitter*1000:.2f} ms")
    print(f"   Jitter Variance (Std Dev): {std_jitter*1000:.2f} ms")
    # Low jitter (< 20-30ms) is good for rhythm games.
    if mean_jitter < 0.03: print("   -> LOW jitter. (Tight timing)")
    elif mean_jitter < 0.06: print("   -> MODERATE jitter. (Playable but loose)")
    else: print("   -> HIGH jitter. (Rhythmically indistinct)")
    
    print("\n3. Pattern Distribution")
    total = sum(patterns.values())
    print(f"   Total Objects: {total}")
    print(f"   Singles: {patterns['single']} ({patterns['single']/total*100:.1f}%)")
    print(f"   Jumps (2 notes): {patterns['jump']} ({patterns['jump']/total*100:.1f}%)")
    print(f"   Hands (3+ notes): {patterns['hand']} ({patterns['hand']/total*100:.1f}%)")
    print(f"   Mines: {patterns['mine']}")
    
    print("\n4. Physical Feasibility")
    print(f"   Impossible Moves Detected: {impossible}")
    if impossible == 0:
        print("   -> PASS. No physically impossible moves found.")
    else:
        print("   -> FAIL. Found impossible moves for standard play:")
        for issue in issues[:5]: # Show first 5
            print(f"      - {issue}")
        if len(issues) > 5: print(f"      ... and {len(issues)-5} more.")

if __name__ == "__main__":
    main()
