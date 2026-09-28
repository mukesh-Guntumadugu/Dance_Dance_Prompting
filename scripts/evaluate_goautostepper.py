import os
import glob
import re
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.spatial.distance import cdist

def parse_sm(sm_path):
    with open(sm_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Get Offset
    offset_match = re.search(r'#OFFSET:([^;]+);', content)
    offset = float(offset_match.group(1)) if offset_match else 0.0

    # Get BPMS (Assuming single BPM for simplicity, goAutoStepper usually uses constant BPM)
    bpms_match = re.search(r'#BPMS:([^;]+);', content)
    bpm_val = 120.0
    if bpms_match:
        bpm_str = bpms_match.group(1).split(',')[0] # Get first BPM
        if '=' in bpm_str:
            bpm_val = float(bpm_str.split('=')[1])

    beat_duration = 60.0 / bpm_val if bpm_val > 0 else 0.5

    # Find all NOTES sections
    notes_sections = re.findall(r'#NOTES:(.*?);', content, re.DOTALL)
    
    charts = {}
    for section in notes_sections:
        lines = [line.strip() for line in section.split('\n') if line.strip()]
        if len(lines) < 6:
            continue
        
        difficulty = lines[2].strip(':')
        
        # The chart data starts from index 5
        chart_data_str = '\n'.join(lines[5:])
        measures = chart_data_str.split(',')
        
        step_times = []
        patterns = []
        
        current_beat = 0.0
        for measure in measures:
            measure_lines = [line.strip() for line in measure.split('\n') if line.strip() and not line.startswith('//')]
            num_lines = len(measure_lines)
            if num_lines == 0:
                continue
            
            beat_increment = 4.0 / num_lines # 4 beats per measure
            
            for line in measure_lines:
                # '0' is empty, '1' is tap, '2' is hold head, '3' is hold tail
                if any(c in '124' for c in line): # 1, 2, 4 are actionable steps
                    time_in_sec = (current_beat * beat_duration) - offset
                    step_times.append(time_in_sec)
                    patterns.append(line)
                
                current_beat += beat_increment
                
        charts[difficulty] = {
            'step_times': np.array(step_times),
            'patterns': patterns
        }
    
    return charts

def evaluate_directory(base_dir):
    results = []
    
    folders = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # We will limit to 50 for speed during this initial test, or user can run it all. Let's do all.
    for folder in tqdm(folders, desc="Evaluating goAutoStepper"):
        sm_files = glob.glob(os.path.join(folder, '*.sm'))
        audio_files = glob.glob(os.path.join(folder, '*.mp3')) + glob.glob(os.path.join(folder, '*.ogg'))
        
        if not sm_files or not audio_files:
            continue
            
        sm_path = sm_files[0]
        audio_path = audio_files[0]
        song_name = os.path.basename(folder)
        
        # 1. Parse SM
        charts = parse_sm(sm_path)
        
        if not charts:
            continue
            
        # 2. Extract Audio Onsets
        try:
            # We use a lower sampling rate and mono for faster onset detection
            y, sr = librosa.load(audio_path, sr=22050, mono=True)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            audio_onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, units='time')
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            continue
            
        # 3. Evaluate each difficulty
        for difficulty, data in charts.items():
            step_times = data['step_times']
            patterns = data['patterns']
            
            total_steps = len(step_times)
            if total_steps == 0:
                continue
                
            unique_patterns = len(set(patterns))
            
            # Onset Alignment (Percentage of steps within 50ms of an audio onset)
            if len(audio_onsets) > 0:
                distances = cdist(step_times.reshape(-1, 1), audio_onsets.reshape(-1, 1), metric='cityblock')
                min_distances = distances.min(axis=1)
                aligned_50ms = np.sum(min_distances <= 0.05) / total_steps
                mean_offset = np.mean(min_distances)
            else:
                aligned_50ms = 0.0
                mean_offset = 0.0
                
            results.append({
                'Song': song_name,
                'Difficulty': difficulty,
                'Total_Steps': total_steps,
                'Unique_Patterns': unique_patterns,
                'Aligned_50ms': aligned_50ms,
                'Mean_Onset_Error': mean_offset
            })
            
    df = pd.DataFrame(results)
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        base_dir = os.path.abspath(sys.argv[1])
    else:
        # Default to the cluster path if no arg is given
        base_dir = os.path.expanduser("~/Beatmap_Generation_DDR/out_GoautoStepper")
        
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        exit(1)
        
    df = evaluate_directory(base_dir)
    
    if len(df) > 0:
        output_csv = "goautostepper_evaluation_results.csv"
        df.to_csv(output_csv, index=False)
        print(f"\\n--- goAutoStepper Evaluation Summary ---")
        
        print("\\nLevel Distribution (Total count of generated difficulties):")
        level_counts = df['Difficulty'].value_counts()
        print(level_counts)
        
        print("\\nMetrics per Difficulty:")
        summary = df.groupby('Difficulty').agg({
            'Total_Steps': 'mean',
            'Unique_Patterns': 'mean',
            'Aligned_50ms': 'mean',
            'Mean_Onset_Error': 'mean'
        }).round(3)
        print(summary)
        
        print("\\nOverall Averages:")
        print(f"Total Charts Analyzed: {len(df)}")
        print(f"Mean Steps per Chart: {df['Total_Steps'].mean():.2f}")
        print(f"Mean Unique Patterns: {df['Unique_Patterns'].mean():.2f}")
        print(f"Onset Alignment (within 50ms): {df['Aligned_50ms'].mean():.2%}")
        print(f"Mean Absolute Onset Error: {df['Mean_Onset_Error'].mean():.4f}s")
        print(f"Results saved to {output_csv}")
    else:
        print("No charts processed.")
