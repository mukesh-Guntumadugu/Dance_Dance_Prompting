#!/usr/bin/env python3
"""
evaluate_model_ssc.py
Evaluates an AI-generated .ssc file against a human ground-truth .ssc file.
Calculates:
1. Rhythmic F1-Score (Timing Match)
2. Cluster Similarity (Style/Physical step distribution match)
3. NPS Error (Density Match)
"""

import os
import argparse
import numpy as np

def parse_ssc_notes(file_path):
    """
    Parses an .ssc file and extracts the notes for each difficulty.
    Returns a dictionary: { difficulty_name: [list of note lines] }
    """
    charts = {}
    if not os.path.exists(file_path):
        return charts

    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    # Split by #NOTEDATA:
    sections = content.split('#NOTEDATA:;')
    
    for section in sections[1:]: # Skip header
        lines = section.split('\n')
        diff_name = "Unknown"
        notes = []
        in_notes = False
        
        for line in lines:
            line = line.strip()
            if line.startswith('#DIFFICULTY:'):
                diff_name = line.split(':')[1].replace(';', '').strip().capitalize()
            elif line.startswith('#NOTES:'):
                in_notes = True
            elif in_notes:
                if line.startswith(';'):
                    break
                if line != '' and not line.startswith('//'):
                    if ',' in line:
                        continue # Measure separator
                    # It's a note line (e.g., "0100" or "0000")
                    notes.append(line)
        
        if diff_name != "Unknown" and notes:
            charts[diff_name] = notes
            
    return charts

def calculate_metrics(human_notes, ai_notes):
    """
    Compares two lists of note lines and returns evaluation metrics.
    """
    # 1. Physical Counts (Style)
    def count_style(notes):
        jumps = 0
        holds = 0
        steps = 0
        active_lines = 0
        for row in notes:
            active_panels = sum(1 for char in row if char in ['1', '2', '4'])
            if active_panels > 0:
                active_lines += 1
            if active_panels >= 2:
                jumps += 1
            if '2' in row:
                holds += 1
            if active_panels == 1:
                steps += 1
        return np.array([steps, jumps, holds]), active_lines
        
    human_vec, human_active = count_style(human_notes)
    ai_vec, ai_active = count_style(ai_notes)
    
    # Cosine Similarity for Style
    norm_human = np.linalg.norm(human_vec)
    norm_ai = np.linalg.norm(ai_vec)
    if norm_human == 0 or norm_ai == 0:
        style_sim = 0.0
    else:
        style_sim = np.dot(human_vec, ai_vec) / (norm_human * norm_ai)
        
    # 2. Rhythmic F1 Score
    # Convert notes to binary "Is there a step on this line?"
    # We truncate to the shortest sequence to compare aligned timing
    min_len = min(len(human_notes), len(ai_notes))
    if min_len == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    h_bin = np.array([1 if sum(1 for c in row if c in ['1','2','4']) > 0 else 0 for row in human_notes[:min_len]])
    a_bin = np.array([1 if sum(1 for c in row if c in ['1','2','4']) > 0 else 0 for row in ai_notes[:min_len]])
    
    true_positives = np.sum((h_bin == 1) & (a_bin == 1))
    false_positives = np.sum((h_bin == 0) & (a_bin == 1))
    false_negatives = np.sum((h_bin == 1) & (a_bin == 0))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # 3. NPS Density
    # Assuming standard 4-beat measures, we just look at total active lines relative to length
    h_density = human_active / len(human_notes)
    a_density = ai_active / len(ai_notes)
    density_error = abs(h_density - a_density)
    
    return f1_score, style_sim, density_error, h_density

def main():
    parser = argparse.ArgumentParser(description='Evaluate AI Beatmap Directory')
    parser.add_argument('--human_dir', type=str, required=True, help='Path to original human .ssc directory')
    parser.add_argument('--ai_dir', type=str, required=True, help='Path to AI generated .ssc directory')
    args = parser.parse_args()
    
    print(f"\n🚀 Batch Evaluating AI Models...")
    print(f"Human Directory: {args.human_dir}")
    print(f"AI Directory:    {args.ai_dir}\n")
    
    # Find all AI ssc files
    ai_files = []
    for root, _, files in os.walk(args.ai_dir):
        for f in files:
            if f.endswith('.ssc') or f.endswith('.sm'):
                ai_files.append(os.path.join(root, f))
                
    if not ai_files:
        print("Could not find any .ssc files in AI directory!")
        return

    # Find all Human ssc files for quick lookup
    human_files_map = {}
    for root, _, files in os.walk(args.human_dir):
        for f in files:
            if f.endswith('.ssc') or f.endswith('.sm'):
                human_files_map[f] = os.path.join(root, f)

    totals = {'f1': [], 'style': [], 'density': []}
    
    for ai_path in ai_files:
        filename = os.path.basename(ai_path)
        human_path = None
        import re
        def clean_name(s):
            return re.sub(r'[^A-Za-z0-9_-]', '_', s.replace('.sm', '').replace('.ssc', ''))
            
        for h_file, h_path in human_files_map.items():
            if clean_name(h_file) in filename:
                human_path = h_path
                break
                
        if not human_path:
            continue
            
        human_charts = parse_ssc_notes(human_path)
        ai_charts = parse_ssc_notes(ai_path)
        
        for diff in ['Beginner', 'Easy', 'Medium', 'Hard', 'Challenge']:
            if diff in human_charts and diff in ai_charts:
                f1, style, d_err, h_den = calculate_metrics(human_charts[diff], ai_charts[diff])
                totals['f1'].append(f1)
                totals['style'].append(style)
                totals['density'].append(d_err)

    if not totals['f1']:
        print("Error: Could not match any AI files to Human files with matching difficulties.")
        return

    avg_f1 = np.mean(totals['f1']) * 100
    avg_style = np.mean(totals['style']) * 100
    avg_density = np.mean(totals['density'])

    print(f"==================================================")
    print(f"📊 FINAL MODEL EVALUATION RESULTS")
    print(f"==================================================")
    print(f"Total Charts Evaluated : {len(totals['f1'])}")
    print(f"Rhythmic F1 (Timing)   : {avg_f1:.1f}%")
    print(f"Cluster Sim (Style)    : {avg_style:.1f}%")
    print(f"Average NPS Error      : {avg_density:.3f} NPS")
    print(f"==================================================")

if __name__ == "__main__":
    main()
